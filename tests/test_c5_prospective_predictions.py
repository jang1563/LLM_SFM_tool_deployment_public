import csv
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from c5_antibody_ood.af3_preflight import (
    ATTESTATION_SCHEMA,
    REQUIRED_ATTESTATION_COMPONENTS,
)
from c5_antibody_ood.af3_phase_outputs import validate_inference_output
from c5_antibody_ood.manifest import (
    load_c5_manifest,
    write_c5_manifest,
)
from c5_antibody_ood.prospective_inputs import _safe_job_name
from c5_antibody_ood.prospective_panel import AF3_COMMIT, canonical_sha256
from c5_antibody_ood.prospective_predictions import (
    ProspectivePredictionError,
    build_prediction_lock,
    collect_af3_predictions,
    validate_prediction_lock,
)
from c5_antibody_ood.source_pilot import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood/c5_prospective_panel_preregistration_v1.json"
)
INPUT_FREEZE = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json"
)
RETAINED_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl"
)
SBATCH = ROOT / "c5_antibody_ood/run_c5_af3_cayuga.sbatch"


def _summary(
    *,
    chain_count: int,
    ranking_score: float,
    iptm: float,
    ptm: float,
    fraction_disordered: float = 0.0,
    has_clash: bool = False,
) -> dict:
    return {
        "ptm": ptm,
        "iptm": iptm,
        "ranking_score": ranking_score,
        "fraction_disordered": fraction_disordered,
        "has_clash": float(has_clash),
        "chain_pair_pae_min": [
            [1.0 for _ in range(chain_count)]
            for _ in range(chain_count)
        ],
        "chain_pair_iptm": [
            [iptm for _ in range(chain_count)]
            for _ in range(chain_count)
        ],
        "chain_ptm": [ptm for _ in range(chain_count)],
        "chain_iptm": [iptm for _ in range(chain_count)],
    }


def _default_scores() -> list[tuple[float, float, float]]:
    return [
        (0.60, 0.60, 0.60),
        (0.65, 0.65, 0.65),
        (0.70, 0.70, 0.70),
        (0.75, 0.75, 0.75),
        (0.80, 0.80, 0.80),
    ]


def _write_job(
    output_root: Path,
    row: dict,
    *,
    scores: list[tuple[float, float, float]] | None = None,
) -> Path:
    scores = scores or _default_scores()
    job_name = _safe_job_name(row["instance_id"])
    job_dir = output_root / job_name
    job_dir.mkdir(parents=True)
    chain_count = len(row["chain_role_mapping"])
    ranking_rows = []
    for sample_index, (ranking_score, iptm, ptm) in enumerate(scores):
        output_id = f"seed-20260725_sample-{sample_index}"
        sample_dir = job_dir / output_id
        sample_dir.mkdir()
        prefix = f"{job_name}_{output_id}"
        (sample_dir / f"{prefix}_model.cif").write_text(
            f"data_{prefix}\n"
        )
        (sample_dir / f"{prefix}_confidences.json").write_text("{}\n")
        (sample_dir / f"{prefix}_summary_confidences.json").write_text(
            json.dumps(
                _summary(
                    chain_count=chain_count,
                    ranking_score=ranking_score,
                    iptm=iptm,
                    ptm=ptm,
                )
            )
            + "\n"
        )
        ranking_rows.append(
            {
                "seed": 20260725,
                "sample": sample_index,
                "ranking_score": ranking_score,
            }
        )

    top_score, top_iptm, top_ptm = max(
        scores,
        key=lambda score: score[0],
    )
    (job_dir / f"{job_name}_model.cif").write_text(f"data_{job_name}\n")
    (job_dir / f"{job_name}_confidences.json").write_text("{}\n")
    (job_dir / f"{job_name}_summary_confidences.json").write_text(
        json.dumps(
            _summary(
                chain_count=chain_count,
                ranking_score=top_score,
                iptm=top_iptm,
                ptm=top_ptm,
            )
        )
        + "\n"
    )
    (job_dir / f"{job_name}_data.json").write_text('{"private":"input"}\n')
    with (job_dir / f"{job_name}_ranking_scores.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "sample", "ranking_score"],
        )
        writer.writeheader()
        writer.writerows(ranking_rows)
    (job_dir / "TERMS_OF_USE.md").write_text("fixture terms\n")
    return job_dir


def test_output_intake_uses_score_iptm_then_lexical_tie_break(tmp_path):
    preregistration = json.loads(PREREGISTRATION.read_text())
    row = load_c5_manifest(RETAINED_MANIFEST)[0]
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    scores = [
        (0.80, 0.80, 0.80),
        (0.80, 0.90, 0.40),
        (0.80, 0.90, 0.40),
        (0.75, 0.75, 0.75),
        (0.70, 0.70, 0.70),
    ]
    _write_job(output_root, row, scores=scores)

    targets = collect_af3_predictions([row], preregistration, output_root)

    assert len(targets) == 1
    assert targets[0]["selected_output_id"] == "seed-20260725_sample-1"
    assert len(targets[0]["samples"]) == 5
    assert targets[0]["job_artifact_set_sha256"]

    split_result = validate_inference_output(
        retained_rows=[row],
        preregistration=preregistration,
        job_name=_safe_job_name(row["instance_id"]),
        job_dir=output_root / _safe_job_name(row["instance_id"]),
    )
    assert split_result["phase"] == "inference"
    assert split_result["verified"] is True
    assert split_result["samples"] == 5
    assert all(split_result["release_boundary"].values()) is False


def test_output_intake_rejects_missing_sample_and_old_nested_layout(tmp_path):
    preregistration = json.loads(PREREGISTRATION.read_text())
    rows = load_c5_manifest(RETAINED_MANIFEST)

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    job_dir = _write_job(missing_root, rows[0])
    sample_dir = job_dir / "seed-20260725_sample-4"
    for path in sample_dir.iterdir():
        path.unlink()
    sample_dir.rmdir()
    with pytest.raises(
        ProspectivePredictionError,
        match="af3_job_entry_set_mismatch",
    ):
        collect_af3_predictions([rows[0]], preregistration, missing_root)

    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    job_dir = _write_job(nested_root, rows[1])
    nested = job_dir / _safe_job_name(rows[1]["instance_id"])
    nested.mkdir()
    with pytest.raises(
        ProspectivePredictionError,
        match="af3_job_entry_set_mismatch",
    ):
        collect_af3_predictions([rows[1]], preregistration, nested_root)


def test_output_intake_rejects_ranking_csv_summary_drift(tmp_path):
    preregistration = json.loads(PREREGISTRATION.read_text())
    row = load_c5_manifest(RETAINED_MANIFEST)[0]
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    job_dir = _write_job(output_root, row)
    job_name = _safe_job_name(row["instance_id"])
    ranking_path = job_dir / f"{job_name}_ranking_scores.csv"
    ranking_path.write_text(
        ranking_path.read_text().replace("20260725,2,0.7", "20260725,2,0.72")
    )

    with pytest.raises(
        ProspectivePredictionError,
        match="ranking_score_summary_csv_mismatch",
    ):
        collect_af3_predictions([row], preregistration, output_root)


def test_output_intake_rejects_symlinked_artifact(tmp_path):
    preregistration = json.loads(PREREGISTRATION.read_text())
    row = load_c5_manifest(RETAINED_MANIFEST)[0]
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    job_dir = _write_job(output_root, row)
    job_name = _safe_job_name(row["instance_id"])
    output_id = "seed-20260725_sample-0"
    model = (
        job_dir
        / output_id
        / f"{job_name}_{output_id}_model.cif"
    )
    external = tmp_path / "external.cif"
    external.write_text("data_external\n")
    model.unlink()
    model.symlink_to(external)

    with pytest.raises(
        ProspectivePredictionError,
        match="af3_sample_contains_symlink",
    ):
        collect_af3_predictions([row], preregistration, output_root)


def test_prediction_lock_is_attested_complete_and_public_projection_is_compact(
    tmp_path,
):
    preregistration = json.loads(PREREGISTRATION.read_text())
    input_freeze = deepcopy(json.loads(INPUT_FREEZE.read_text()))
    rows = load_c5_manifest(RETAINED_MANIFEST)
    retained_manifest = tmp_path / "retained.jsonl"
    write_c5_manifest(retained_manifest, rows)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_hashes = {}
    for row in rows:
        path = input_dir / f"{_safe_job_name(row['instance_id'])}.json"
        path.write_text('{"dialect":"alphafold3"}\n')
        input_hashes[row["target_id"]] = sha256_file(path)
    input_set_sha256 = canonical_sha256(input_hashes)
    input_freeze["af3_inputs"]["files"] = len(rows)
    input_freeze["af3_inputs"]["af3_input_set_sha256"] = input_set_sha256

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=benchmark_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.org"],
        cwd=benchmark_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=benchmark_dir,
        check=True,
    )
    (benchmark_dir / "README").write_text("benchmark fixture\n")
    subprocess.run(["git", "add", "README"], cwd=benchmark_dir, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=benchmark_dir,
        check=True,
    )
    benchmark_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=benchmark_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    attestation = {
        "schema_version": ATTESTATION_SCHEMA,
        "ready_for_af3_prediction": True,
        "violations": [],
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        "components": {
            name: True for name in REQUIRED_ATTESTATION_COMPONENTS
        },
        "checksums": {
            "benchmark_commit": benchmark_commit,
            "source_commit": AF3_COMMIT,
            "container_sha256": "1" * 64,
            "model_parameter_set_sha256": "2" * 64,
            "model_manifest_sha256": "3" * 64,
            "database_manifest_sha256": "4" * 64,
            "retained_manifest_sha256": "5" * 64,
            "af3_input_set_sha256": input_set_sha256,
        },
    }
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True))

    output_root = tmp_path / "outputs"
    output_root.mkdir()
    for row in rows:
        _write_job(output_root, row)

    private_lock, public_freeze = build_prediction_lock(
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=rows,
        retained_manifest_path=retained_manifest,
        private_input_dir=input_dir,
        attestation_path=attestation_path,
        expected_attestation_sha256=sha256_file(attestation_path),
        output_root=output_root,
        benchmark_dir=benchmark_dir,
        created_at_utc="2026-07-25T12:00:00+00:00",
    )

    assert private_lock["counts"] == {
        "targets": 120,
        "targets_by_role": {
            "calibration": 80,
            "evaluation": 40,
        },
        "samples_per_target": 5,
        "samples": 600,
    }
    assert private_lock["evidence_boundary"] == {
        "dockq_or_native_interface_labels_read": False,
        "evaluation_labels_read": False,
        "label_input_accepted_by_this_command": False,
    }
    assert public_freeze["validation"][
        "complete_five_sample_set_per_target"
    ] is True
    assert public_freeze["decision"][
        "ready_for_calibration_label_reveal"
    ] is True
    assert public_freeze["decision"][
        "ready_for_evaluation_label_reveal"
    ] is False
    rendered_public = json.dumps(public_freeze, sort_keys=True)
    assert rows[0]["target_id"] not in rendered_public
    assert rows[0]["instance_id"] not in rendered_public
    assert "relative_path" not in rendered_public
    assert '"targets": [' not in rendered_public

    mutated_selection = deepcopy(private_lock)
    mutated_selection["targets"][0]["selected_output_id"] = (
        "seed-20260725_sample-0"
    )
    assert "selected_output_rule_mismatch" in validate_prediction_lock(
        mutated_selection,
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=rows,
    )
    private_lock["targets"][0]["dockq"] = 0.99
    assert "prediction_lock_contains_label_key" in validate_prediction_lock(
        private_lock,
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=rows,
    )


def test_cayuga_array_passes_common_output_root_to_official_af3_writer():
    script = SBATCH.read_text()

    assert '--output_dir=/root/af_output' in script
    assert '--output_dir="/root/af_output/${JOB_NAME}"' not in script
    assert 'TARGET_OUTPUT="${AF3_OUTPUT_DIR}/${JOB_NAME}"' in script
