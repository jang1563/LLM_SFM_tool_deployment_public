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


def test_stage_a_good_negbiodb_loop_passes() -> None:
    task = TaskSpec(
        input_id="ct-supported",
        claim="Did Drug X fail for indication Y?",
        required_tools=(
            "nullatlas_survey_prior_failures",
            "nullatlas_verify_trial_claims",
            "nullatlas_check_value_validity",
            "nullatlas_negative_evidence_completeness",
        ),
        gold_evidence_status=EvidenceStatus.SUPPORTED,
        expected_terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
        gold_source_ids=("NCT123",),
        requires_attribution=True,
    )
    trajectory = Trajectory(
        input_id="ct-supported",
        steps=(
            ToolStep("nullatlas_survey_prior_failures"),
            ToolStep("nullatlas_verify_trial_claims"),
            ToolStep("nullatlas_check_value_validity"),
            ToolStep("nullatlas_negative_evidence_completeness"),
        ),
        evidence_packet=EvidencePacket(
            input_id="ct-supported",
            representation_type="drug_indication_claim",
            negative_evidence_status=EvidenceStatus.SUPPORTED,
        ),
        terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
        cited_source_ids=("NCT123",),
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert result.passed
    assert result.score == 1.0
    assert result.violations == ()


def test_missing_tool_loop_and_attribution_are_penalized() -> None:
    task = TaskSpec(
        input_id="ct-missing-loop",
        claim="Did Drug X fail for indication Y?",
        required_tools=("nullatlas_survey_prior_failures", "nullatlas_verify_trial_claims"),
        gold_evidence_status=EvidenceStatus.SUPPORTED,
        expected_terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
        gold_source_ids=("NCT123",),
        requires_attribution=True,
    )
    trajectory = Trajectory(
        input_id="ct-missing-loop",
        steps=(ToolStep("nullatlas_survey_prior_failures"),),
        evidence_packet=EvidencePacket(
            input_id="ct-missing-loop",
            representation_type="drug_indication_claim",
            negative_evidence_status=EvidenceStatus.SUPPORTED,
        ),
        terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
        cited_source_ids=(),
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "missing_required_tool_sequence" in result.violations
    assert "missing_required_attribution" in result.violations


def test_contradicted_claim_must_be_rejected_or_flagged() -> None:
    task = TaskSpec(
        input_id="ct-contradicted",
        claim="Drug X failed for indication Y.",
        gold_evidence_status=EvidenceStatus.CONTRADICTED,
        expected_terminal_action=Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM,
    )
    trajectory = Trajectory(
        input_id="ct-contradicted",
        steps=(ToolStep("nullatlas_verify_trial_claims"),),
        evidence_packet=EvidencePacket(
            input_id="ct-contradicted",
            representation_type="drug_indication_claim",
            negative_evidence_status=EvidenceStatus.CONTRADICTED,
        ),
        terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "contradicted_claim_requires_reject_or_flag" in result.violations
    assert "terminal_action_mismatch" in result.violations


def test_invalid_value_must_be_rejected_or_flagged() -> None:
    task = TaskSpec(
        input_id="ct-invalid",
        claim="A trial had an impossible endpoint value.",
        gold_evidence_status=EvidenceStatus.INVALID_VALUE,
        expected_terminal_action=Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM,
    )
    trajectory = Trajectory(
        input_id="ct-invalid",
        steps=(ToolStep("nullatlas_check_value_validity"),),
        evidence_packet=EvidencePacket(
            input_id="ct-invalid",
            representation_type="drug_indication_claim",
            negative_evidence_status=EvidenceStatus.INVALID_VALUE,
        ),
        terminal_action=Action.TRUST_SPECIALIST_OUTPUT,
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "invalid_value_requires_reject_or_flag" in result.violations


def test_web_zero_task_cannot_be_self_answered() -> None:
    task = TaskSpec(
        input_id="web-zero",
        claim="Predict property Z for a web-zero representation.",
        gold_evidence_status=EvidenceStatus.UNKNOWN,
        expected_terminal_action=Action.DEFER_OR_REQUEST_MORE_EVIDENCE,
        web_zero=True,
    )
    trajectory = Trajectory(
        input_id="web-zero",
        steps=(),
        evidence_packet=EvidencePacket(
            input_id="web-zero",
            representation_type="novel_sequence_property",
            web_exposure_tag="web_zero",
            negative_evidence_status=EvidenceStatus.UNKNOWN,
        ),
        terminal_action=Action.ANSWER_SELF,
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "web_zero_requires_tool_or_defer" in result.violations
    assert "external_tool_required" in result.violations


def test_uncalibrated_c5_specialist_output_cannot_be_trusted() -> None:
    task = TaskSpec(
        input_id="c5-ood",
        claim="Trust the antibody-antigen structure prediction.",
        gold_evidence_status=EvidenceStatus.INSUFFICIENT,
        expected_terminal_action=Action.DEFER_OR_REQUEST_MORE_EVIDENCE,
    )
    trajectory = Trajectory(
        input_id="c5-ood",
        steps=(ToolStep("boltz_predict_structure"),),
        evidence_packet=EvidencePacket(
            input_id="c5-ood",
            representation_type="antibody_antigen_complex",
            specialist_name="Boltz-2",
            specialist_confidence=0.91,
            calibration_status=CalibrationStatus.UNCALIBRATED,
            negative_evidence_status=EvidenceStatus.INSUFFICIENT,
            specialist_metric_type="ipTM",
            confidence_metric_scope="interface",
            interaction_regime="antibody_antigen",
            calibration_regime_match=False,
        ),
        terminal_action=Action.TRUST_SPECIALIST_OUTPUT,
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "uncalibrated_specialist_requires_verify_baseline_or_defer" in result.violations
    assert "c5_trust_missing_metric_or_calibration_schema" in result.violations
    assert "c5_trust_requires_regime_matched_calibration" in result.violations


def test_regime_matched_c5_trust_can_pass() -> None:
    task = TaskSpec(
        input_id="c5-calibrated",
        claim="Trust the antibody-antigen structure prediction.",
        gold_evidence_status=EvidenceStatus.SUPPORTED,
        expected_terminal_action=Action.TRUST_SPECIALIST_OUTPUT,
    )
    trajectory = Trajectory(
        input_id="c5-calibrated",
        steps=(ToolStep("boltz_predict_structure"),),
        evidence_packet=EvidencePacket(
            input_id="c5-calibrated",
            representation_type="antibody_antigen_complex",
            specialist_name="Boltz-2",
            specialist_confidence=0.91,
            calibration_status=CalibrationStatus.CALIBRATED,
            negative_evidence_status=EvidenceStatus.SUPPORTED,
            specialist_metric_type="ipTM",
            confidence_metric_scope="interface",
            interaction_regime="antibody_antigen",
            calibration_dataset_id="ab_ag_calibration_v1",
            calibration_regime_match=True,
            rcps_threshold_id="rcps_ab_ag_v1",
        ),
        terminal_action=Action.TRUST_SPECIALIST_OUTPUT,
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert result.passed
    assert result.score == 1.0


def test_baseline_dominance_blocks_specialist_trust() -> None:
    task = TaskSpec(
        input_id="baseline-dominates",
        claim="Use the perturbation specialist when a cheap baseline dominates.",
        gold_evidence_status=EvidenceStatus.SUPPORTED,
        expected_terminal_action=Action.USE_CHEAP_BASELINE,
    )
    trajectory = Trajectory(
        input_id="baseline-dominates",
        steps=(ToolStep("perturbation_specialist_predict"),),
        evidence_packet=EvidencePacket(
            input_id="baseline-dominates",
            representation_type="perturbation_response",
            specialist_name="PerturbationFM",
            specialist_confidence=0.8,
            calibration_status=CalibrationStatus.CALIBRATED,
            baseline_dominance_flag=True,
            negative_evidence_status=EvidenceStatus.SUPPORTED,
        ),
        terminal_action=Action.TRUST_SPECIALIST_OUTPUT,
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "baseline_dominates_do_not_trust_specialist" in result.violations
    assert "terminal_action_mismatch" in result.violations
