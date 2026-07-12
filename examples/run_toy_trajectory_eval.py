"""Run a toy trajectory evaluation without any model/API calls."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import (
    Action,
    CalibrationStatus,
    EvidencePacket,
    EvidenceStatus,
    TaskSpec,
    ToolStep,
    Trajectory,
    TrajectoryEvaluator,
)


def main() -> None:
    evaluator = TrajectoryEvaluator()

    task = TaskSpec(
        input_id="ct-001",
        claim="Did Drug X fail for indication Y?",
        required_tools=(
            "nullatlas_survey_prior_failures",
            "nullatlas_verify_trial_claims",
            "nullatlas_check_value_validity",
            "nullatlas_negative_evidence_completeness",
        ),
        gold_evidence_status=EvidenceStatus.SUPPORTED,
        expected_terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
        gold_source_ids=("NCT00000001",),
        requires_attribution=True,
    )

    good_trajectory = Trajectory(
        input_id="ct-001",
        steps=(
            ToolStep("nullatlas_survey_prior_failures"),
            ToolStep("nullatlas_verify_trial_claims"),
            ToolStep("nullatlas_check_value_validity"),
            ToolStep("nullatlas_negative_evidence_completeness"),
        ),
        evidence_packet=EvidencePacket(
            input_id="ct-001",
            representation_type="drug_indication_claim",
            calibration_status=CalibrationStatus.NOT_APPLICABLE,
            negative_evidence_status=EvidenceStatus.SUPPORTED,
        ),
        terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
        cited_source_ids=("NCT00000001",),
    )

    bad_trajectory = Trajectory(
        input_id="ct-001",
        steps=(),
        evidence_packet=EvidencePacket(
            input_id="ct-001",
            representation_type="drug_indication_claim",
            calibration_status=CalibrationStatus.UNKNOWN,
            negative_evidence_status=EvidenceStatus.UNKNOWN,
        ),
        terminal_action=Action.ANSWER_SELF,
        cited_source_ids=(),
    )

    for name, trajectory in (
        ("good", good_trajectory),
        ("bad", bad_trajectory),
    ):
        result = evaluator.evaluate(task, trajectory)
        print(f"{name}: score={result.score:.3f} passed={result.passed}")
        if result.violations:
            print("  violations:", ", ".join(result.violations))


if __name__ == "__main__":
    main()
