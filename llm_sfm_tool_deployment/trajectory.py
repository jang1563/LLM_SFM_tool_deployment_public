"""Trajectory schema and deterministic evaluator.

The first implementation target is deliberately small: score whether an agent
uses the right scientific tool loop and obeys hard deployment policy before we
claim any SFT, DPO, or RLVR improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class Action(str, Enum):
    """Terminal or high-level orchestration actions."""

    ANSWER_SELF = "answer_self"
    CALL_SPECIALIST_TOOL = "call_specialist_tool"
    GROUND_WITH_ATTRIBUTION = "ground_with_attribution"
    TRUST_SPECIALIST_OUTPUT = "trust_specialist_output"
    VERIFY_WITH_ASSAY_OR_DATABASE = "verify_with_assay_or_database"
    USE_CHEAP_BASELINE = "use_cheap_baseline"
    DEFER_OR_REQUEST_MORE_EVIDENCE = "defer_or_request_more_evidence"
    REJECT_OR_FLAG_UNSUPPORTED_CLAIM = "reject_or_flag_unsupported_claim"


class EvidenceStatus(str, Enum):
    """Gold or predicted evidence state for a biological claim."""

    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONTRADICTED = "contradicted"
    INVALID_VALUE = "invalid_value"
    UNKNOWN = "unknown"


class CalibrationStatus(str, Enum):
    """Whether a specialist confidence score is deployable for this regime."""

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


DEFAULT_ALLOWED_ACTIONS = (
    Action.ANSWER_SELF,
    Action.CALL_SPECIALIST_TOOL,
    Action.GROUND_WITH_ATTRIBUTION,
    Action.TRUST_SPECIALIST_OUTPUT,
    Action.VERIFY_WITH_ASSAY_OR_DATABASE,
    Action.USE_CHEAP_BASELINE,
    Action.DEFER_OR_REQUEST_MORE_EVIDENCE,
    Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM,
)


@dataclass(frozen=True)
class ToolStep:
    """One tool call and the compact observation needed for scoring."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    observation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePacket:
    """Compact evidence packet carried by the trajectory."""

    input_id: str
    representation_type: str
    web_exposure_tag: str = "unknown"
    specialist_name: str | None = None
    specialist_output: Mapping[str, Any] = field(default_factory=dict)
    specialist_confidence: float | None = None
    calibration_status: CalibrationStatus | str = CalibrationStatus.NOT_APPLICABLE
    cheap_baseline_output: Mapping[str, Any] = field(default_factory=dict)
    baseline_dominance_flag: bool = False
    negative_evidence_status: EvidenceStatus | str = EvidenceStatus.UNKNOWN
    claim_guard_status: str = "unknown"
    allowed_actions: Sequence[Action | str] = DEFAULT_ALLOWED_ACTIONS
    hidden_truth_pointer: str | None = None
    specialist_metric_type: str | None = None
    confidence_metric_scope: str | None = None
    interaction_regime: str | None = None
    calibration_dataset_id: str | None = None
    calibration_regime_match: bool | None = None
    rcps_threshold_id: str | None = None
    interface_label_source: str | None = None
    assay_or_structure_truth_status: str | None = None
    fail_closed_reason: str | None = None


@dataclass(frozen=True)
class Trajectory:
    """Model or policy output to be scored."""

    input_id: str
    steps: Sequence[ToolStep]
    evidence_packet: EvidencePacket
    terminal_action: Action | str
    cited_source_ids: Sequence[str] = ()
    predicted_evidence_status: EvidenceStatus | str | None = None
    rationale: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    """Hidden task metadata used by the evaluator, not shown to the model."""

    input_id: str
    claim: str
    required_tools: Sequence[str] = ()
    gold_evidence_status: EvidenceStatus | str = EvidenceStatus.UNKNOWN
    expected_terminal_action: Action | str | None = None
    gold_source_ids: Sequence[str] = ()
    requires_attribution: bool = False
    requires_external_tool: bool = True
    web_zero: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    """Deterministic score plus traceable reasons."""

    input_id: str
    earned: float
    possible: float
    reward_breakdown: Mapping[str, float]
    violations: Sequence[str]

    @property
    def score(self) -> float:
        if self.possible == 0:
            return 0.0
        return self.earned / self.possible

    @property
    def passed(self) -> bool:
        return not self.violations and self.earned == self.possible


class TrajectoryEvaluator:
    """Scores verifiable trajectory slices and hard policy gates."""

    def evaluate(self, task: TaskSpec, trajectory: Trajectory) -> EvaluationResult:
        if task.input_id != trajectory.input_id:
            raise ValueError(
                f"Task id {task.input_id!r} does not match trajectory id "
                f"{trajectory.input_id!r}."
            )

        terminal = self._action(trajectory.terminal_action)
        allowed = {self._action(action) for action in trajectory.evidence_packet.allowed_actions}
        predicted_status = self._evidence_status(
            trajectory.predicted_evidence_status
            or trajectory.evidence_packet.negative_evidence_status
        )
        gold_status = self._evidence_status(task.gold_evidence_status)

        reward: dict[str, float] = {}
        violations: list[str] = []

        reward["action_allowed"] = float(terminal in allowed)
        if terminal not in allowed:
            violations.append("action_not_allowed")

        tool_sequence_ok = self._contains_ordered_tools(
            [step.name for step in trajectory.steps], task.required_tools
        )
        reward["required_tool_sequence"] = float(tool_sequence_ok)
        if task.required_tools and not tool_sequence_ok:
            violations.append("missing_required_tool_sequence")

        reward["evidence_status"] = float(predicted_status == gold_status)
        if predicted_status != gold_status:
            violations.append("evidence_status_mismatch")

        attribution_ok = self._attribution_ok(task, trajectory)
        reward["attribution"] = float(attribution_ok)
        if not attribution_ok:
            violations.append("missing_required_attribution")

        action_ok = (
            True
            if task.expected_terminal_action is None
            else terminal == self._action(task.expected_terminal_action)
        )
        reward["terminal_action"] = float(action_ok)
        if not action_ok:
            violations.append("terminal_action_mismatch")

        policy_violations = self._policy_violations(task, trajectory, terminal, gold_status)
        reward["policy_compliance"] = float(not policy_violations)
        violations.extend(policy_violations)

        possible = float(len(reward))
        earned = float(sum(reward.values()))
        return EvaluationResult(
            input_id=task.input_id,
            earned=earned,
            possible=possible,
            reward_breakdown=reward,
            violations=tuple(violations),
        )

    def _policy_violations(
        self,
        task: TaskSpec,
        trajectory: Trajectory,
        terminal: Action,
        gold_status: EvidenceStatus,
    ) -> list[str]:
        packet = trajectory.evidence_packet
        calibration = self._calibration_status(packet.calibration_status)
        violations: list[str] = []

        if task.web_zero and terminal == Action.ANSWER_SELF:
            violations.append("web_zero_requires_tool_or_defer")

        if task.requires_external_tool and terminal == Action.ANSWER_SELF:
            violations.append("external_tool_required")

        if packet.baseline_dominance_flag and terminal == Action.TRUST_SPECIALIST_OUTPUT:
            violations.append("baseline_dominates_do_not_trust_specialist")

        if (
            terminal == Action.TRUST_SPECIALIST_OUTPUT
            and calibration != CalibrationStatus.CALIBRATED
        ):
            violations.append("uncalibrated_specialist_requires_verify_baseline_or_defer")

        if (
            gold_status == EvidenceStatus.CONTRADICTED
            and terminal != Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM
        ):
            violations.append("contradicted_claim_requires_reject_or_flag")

        if (
            gold_status == EvidenceStatus.INVALID_VALUE
            and terminal != Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM
        ):
            violations.append("invalid_value_requires_reject_or_flag")

        if (
            gold_status == EvidenceStatus.INSUFFICIENT
            and terminal == Action.TRUST_SPECIALIST_OUTPUT
        ):
            violations.append("insufficient_evidence_cannot_be_trusted")

        if terminal == Action.TRUST_SPECIALIST_OUTPUT:
            violations.extend(self._specialist_trust_violations(packet))

        return violations

    def _specialist_trust_violations(self, packet: EvidencePacket) -> list[str]:
        violations: list[str] = []
        regime = (packet.interaction_regime or "").lower()

        if regime == "antibody_antigen":
            required_fields = {
                "specialist_metric_type": packet.specialist_metric_type,
                "confidence_metric_scope": packet.confidence_metric_scope,
                "calibration_dataset_id": packet.calibration_dataset_id,
                "rcps_threshold_id": packet.rcps_threshold_id,
            }
            missing = [name for name, value in required_fields.items() if not value]
            if missing:
                violations.append("c5_trust_missing_metric_or_calibration_schema")
            if packet.calibration_regime_match is not True:
                violations.append("c5_trust_requires_regime_matched_calibration")

        return violations

    def _attribution_ok(self, task: TaskSpec, trajectory: Trajectory) -> bool:
        if not task.requires_attribution and not task.gold_source_ids:
            return True
        cited = set(trajectory.cited_source_ids)
        required = set(task.gold_source_ids)
        return bool(required) and required.issubset(cited)

    def _contains_ordered_tools(
        self, observed_tools: Sequence[str], required_tools: Sequence[str]
    ) -> bool:
        if not required_tools:
            return True

        cursor = 0
        for observed in observed_tools:
            if observed == required_tools[cursor]:
                cursor += 1
                if cursor == len(required_tools):
                    return True
        return False

    def _action(self, value: Action | str) -> Action:
        if isinstance(value, Action):
            return value
        return Action(value)

    def _evidence_status(self, value: EvidenceStatus | str) -> EvidenceStatus:
        if isinstance(value, EvidenceStatus):
            return value
        return EvidenceStatus(value)

    def _calibration_status(self, value: CalibrationStatus | str) -> CalibrationStatus:
        if isinstance(value, CalibrationStatus):
            return value
        return CalibrationStatus(value)
