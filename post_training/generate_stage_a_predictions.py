#!/usr/bin/env python3
"""Generate saved Stage A prediction JSONL files.

This producer is artifact-first: it writes predictions to disk, then the
separate offline scorer evaluates that file. Public modes are deterministic and
no-API. Live API generation is opt-in and requires an explicit safety flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_sft_smoke_eval import load_jsonl


DATASET = "negbiodb_ct_stage_a_saved_predictions_v1"
DEFAULT_RUN_ID = "stage_a_saved_prediction_smoke"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_CONTRACTS = (
    "basic",
    "stage_a_v2_strict",
    "stage_a_v3_tool_trace",
    "stage_a_v4_canonical_json",
)


def effective_model(mode: str, model: str | None) -> str | None:
    if model:
        return model
    if mode == "openai_chat":
        return DEFAULT_OPENAI_MODEL
    if mode == "hf_chat":
        return DEFAULT_HF_MODEL
    return None


def disable_transformers_torchvision_probe(import_utils: Any | None = None) -> bool:
    """Keep text-only causal LM loading from importing broken torchvision builds."""

    if import_utils is None:
        try:
            import transformers.utils.import_utils as import_utils
        except Exception:
            return False

    if getattr(import_utils, "_stage_a_text_only_torchvision_patch", False):
        return True

    original_package_available = import_utils._is_package_available

    def text_only_package_available(
        package_name: str,
        return_version: bool = False,
    ) -> tuple[bool, str | None]:
        if package_name == "torchvision":
            return (False, "N/A") if return_version else (False, None)
        return original_package_available(package_name, return_version)

    import_utils._is_package_available = text_only_package_available
    for name in (
        "is_torchvision_available",
        "is_torchvision_v2_available",
        "is_torchvision_greater_or_equal",
    ):
        cached = getattr(import_utils, name, None)
        if hasattr(cached, "cache_clear"):
            cached.cache_clear()
    import_utils._stage_a_text_only_torchvision_patch = True
    return True


def source_case_id(row: Mapping[str, Any]) -> str:
    for key in ("source_manifest_case_id", "task_id", "case_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("SFT row is missing source_manifest_case_id/task_id/case_id.")


def target_trajectory(row: Mapping[str, Any]) -> Mapping[str, Any]:
    trajectory = row.get("target_trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"{source_case_id(row)} is missing target_trajectory.")
    return trajectory


def prompt_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return only model-visible system/user messages from an SFT row."""

    out: list[dict[str, str]] = []
    for message in row.get("messages", ()):
        role = message.get("role")
        if role == "assistant" and "tool_call" in message:
            break
        if role not in {"system", "user"}:
            continue
        content = message.get("content", "")
        out.append({"role": str(role), "content": content_text(content)})
    if not out:
        raise ValueError(f"{source_case_id(row)} has no model-visible prompt messages.")
    return out


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True)


def prompt_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(prompt_messages(row), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def generation_prompt_hash(row: Mapping[str, Any], *, prompt_contract: str) -> str:
    payload = json.dumps(api_prompt_messages(row, prompt_contract=prompt_contract), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def generate_rows(
    sft_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    run_id: str = DEFAULT_RUN_ID,
    model: str | None = None,
    allow_live_api: bool = False,
    allow_model_load: bool = False,
    device: str = "auto",
    max_new_tokens: int = 512,
    local_files_only: bool = True,
    prompt_contract: str = "basic",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if prompt_contract not in PROMPT_CONTRACTS:
        raise ValueError(f"Unknown prompt_contract: {prompt_contract}")
    selected = list(sft_rows[:limit] if limit is not None else sft_rows)
    model_id = effective_model(mode, model)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        if mode == "oracle":
            rows.append(oracle_prediction(row, run_id=run_id))
        elif mode == "self_answer":
            rows.append(self_answer_prediction(row, run_id=run_id))
        elif mode == "compact_tool_names_oracle":
            rows.append(compact_tool_names_oracle_prediction(row, run_id=run_id))
        elif mode == "openai_chat":
            rows.append(
                openai_chat_prediction(
                    row,
                    run_id=run_id,
                    model=model_id or DEFAULT_OPENAI_MODEL,
                    allow_live_api=allow_live_api,
                    prompt_contract=prompt_contract,
                    index=index,
                    total=len(selected),
                )
            )
        elif mode == "hf_chat":
            client = get_hf_chat_client(
                model=model_id or DEFAULT_HF_MODEL,
                allow_model_load=allow_model_load,
                device=device,
                max_new_tokens=max_new_tokens,
                local_files_only=local_files_only,
            )
            rows.append(
                hf_chat_prediction(
                    row,
                    run_id=run_id,
                    client=client,
                    prompt_contract=prompt_contract,
                    index=index,
                    total=len(selected),
                )
            )
        else:
            raise ValueError(f"Unknown prediction generation mode: {mode}")
    return rows


def base_prediction_row(row: Mapping[str, Any], *, run_id: str, source: str) -> dict[str, Any]:
    return {
        "case_id": source_case_id(row),
        "dataset": DATASET,
        "source": source,
        "run_id": run_id,
        "split": row.get("split"),
        "case_family": row.get("case_family"),
        "prompt_hash": prompt_hash(row),
    }


def oracle_prediction(row: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    out = base_prediction_row(row, run_id=run_id, source="oracle_from_sft_target")
    out["trajectory"] = dict(target_trajectory(row))
    return out


def self_answer_prediction(row: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    out = base_prediction_row(row, run_id=run_id, source="deterministic_self_answer")
    out["prediction"] = {
        "action": "self_answer",
        "evidence_status": "unknown",
        "tool_calls": [],
        "cited_source_ids": [],
        "rationale": "No external evidence packet was generated.",
    }
    return out


def compact_tool_names_oracle_prediction(
    row: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    trajectory = target_trajectory(row)
    out = base_prediction_row(row, run_id=run_id, source="compact_tool_names_oracle")
    out["prediction"] = {
        "action": trajectory.get("terminal_action"),
        "evidence_status": trajectory.get("predicted_evidence_status"),
        "tool_calls": [
            str(step.get("name"))
            for step in trajectory.get("steps", ())
            if isinstance(step, Mapping)
        ],
        "cited_source_ids": list(trajectory.get("cited_source_ids", ())),
        "rationale": "Oracle final decision with compact tool names only.",
    }
    return out


def openai_chat_prediction(
    row: Mapping[str, Any],
    *,
    run_id: str,
    model: str,
    allow_live_api: bool,
    prompt_contract: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    if not allow_live_api:
        raise RuntimeError("openai_chat mode requires --allow-live-api.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("openai_chat mode requires OPENAI_API_KEY.")

    from openai import OpenAI

    client = OpenAI()
    messages = api_prompt_messages(row, prompt_contract=prompt_contract)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=512,
    )
    raw_output = response.choices[0].message.content or ""

    out = base_prediction_row(row, run_id=run_id, source="openai_chat")
    out.update(
        {
            "provider": "openai",
            "model": model,
            "prompt_contract": prompt_contract,
            "generation_prompt_hash": generation_prompt_hash(row, prompt_contract=prompt_contract),
            "raw_output": raw_output,
        }
    )
    print(f"[{index + 1}/{total}] wrote openai_chat prediction for {out['case_id']}", flush=True)
    return out


_HF_CHAT_CLIENTS: dict[tuple[str, str, int, bool], Any] = {}


def get_hf_chat_client(
    *,
    model: str,
    allow_model_load: bool,
    device: str,
    max_new_tokens: int,
    local_files_only: bool,
) -> Any:
    if not allow_model_load:
        raise RuntimeError("hf_chat mode requires --allow-model-load.")
    key = (model, device, max_new_tokens, local_files_only)
    if key not in _HF_CHAT_CLIENTS:
        _HF_CHAT_CLIENTS[key] = HuggingFaceChatClient(
            model,
            device=device,
            max_new_tokens=max_new_tokens,
            local_files_only=local_files_only,
        )
    return _HF_CHAT_CLIENTS[key]


class HuggingFaceChatClient:
    """Small Transformers causal-LM wrapper for Stage A prediction artifacts."""

    def __init__(
        self,
        model: str,
        *,
        device: str,
        max_new_tokens: int,
        local_files_only: bool,
    ) -> None:
        import torch

        disable_transformers_torchvision_probe()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model_id = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            local_files_only=local_files_only,
            torch_dtype="auto",
        )
        self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        import torch

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
            prompt += "\nassistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def hf_chat_prediction(
    row: Mapping[str, Any],
    *,
    run_id: str,
    client: Any,
    prompt_contract: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    raw_output = client.generate(api_prompt_messages(row, prompt_contract=prompt_contract))
    out = base_prediction_row(row, run_id=run_id, source="hf_chat")
    out.update(
        {
            "provider": "huggingface_transformers",
            "model": client.model_id,
            "device": client.device,
            "prompt_contract": prompt_contract,
            "generation_prompt_hash": generation_prompt_hash(row, prompt_contract=prompt_contract),
            "raw_output": raw_output,
        }
    )
    print(f"[{index + 1}/{total}] wrote hf_chat prediction for {out['case_id']}", flush=True)
    return out


def api_prompt_messages(
    row: Mapping[str, Any],
    *,
    prompt_contract: str = "basic",
) -> list[dict[str, str]]:
    if prompt_contract not in PROMPT_CONTRACTS:
        raise ValueError(f"Unknown prompt_contract: {prompt_contract}")
    messages = prompt_messages(row)
    system_suffix = prompt_contract_suffix(prompt_contract)
    out = [dict(message) for message in messages]
    for message in out:
        if message["role"] == "system":
            message["content"] = message["content"] + system_suffix
            return out
    out.insert(0, {"role": "system", "content": system_suffix.strip()})
    return out


def prompt_contract_suffix(prompt_contract: str) -> str:
    basic = (
        "\n\nYou are producing an offline Stage A prediction artifact. "
        "Return JSON only with keys: action, evidence_status, tool_calls, "
        "cited_source_ids, rationale. If you list tool calls, use objects with "
        "name and arguments. Use placeholder arguments "
        "{\"drug_id\":\"<drug_id>\",\"condition_id\":\"<condition_id>\"}. "
        "Do not invent source IDs; cite only IDs you can justify from the prompt."
    )
    if prompt_contract == "basic":
        return basic
    if prompt_contract == "stage_a_v2_strict":
        return basic + (
            "\n\nStrict Stage A output contract:\n"
            "- Output exactly one JSON object and no markdown, prose, or code fence.\n"
            "- action must be exactly one of: ground, reject, defer, verify, flag, self_answer.\n"
            "- evidence_status must be exactly one of: supported, contradicted, invalid_value, insufficient, unknown.\n"
            "- Never use evidence_status values such as sourced, verified, positive, negative, mixed, or related.\n"
            "- tool_calls must be a list of objects, never plain strings.\n"
            "- Each tool call object must use an allowed tool name from the user task.\n"
            "- Each tool call arguments object must include both drug_id and condition_id.\n"
            "- For uncertainty or related but not decisive evidence, prefer action verify or defer with evidence_status insufficient.\n"
            "- cited_source_ids must include only source IDs that are visible in the prompt; use [] if none are visible.\n"
            "- Do not include hidden labels, gold status, expected action, or evaluator metadata."
        )
    if prompt_contract == "stage_a_v3_tool_trace":
        return basic + (
            "\n\nStrict Stage A tool-trace contract:\n"
            "- Output exactly one JSON object and no markdown, prose, or code fence.\n"
            "- action must be exactly one of: ground, reject, defer, verify, flag, self_answer.\n"
            "- evidence_status must be exactly one of: supported, contradicted, invalid_value, insufficient, unknown.\n"
            "- Never use evidence_status values such as sourced, verified, positive, negative, mixed, or related.\n"
            "- tool_calls must be a list of exactly four objects in this order:\n"
            "  1. nullatlas_survey_prior_failures\n"
            "  2. nullatlas_verify_trial_claims\n"
            "  3. nullatlas_check_value_validity\n"
            "  4. nullatlas_negative_evidence_completeness\n"
            "- Each tool call object must have keys name and arguments.\n"
            "- Each arguments object must include both drug_id and condition_id.\n"
            "- Use placeholder argument values \"<drug_id>\" and \"<condition_id>\" when exact IDs are not visible.\n"
            "- cited_source_ids must include only source IDs that are visible in the prompt; use [] if none are visible.\n"
            "- Do not invent NCT IDs, trial IDs, hidden labels, gold status, expected action, case family, source task IDs, or evaluator metadata.\n"
            "- If the prompt does not expose source IDs, prioritize a valid tool trace and honest empty citations over fabricated attribution."
        )
    if prompt_contract == "stage_a_v4_canonical_json":
        return basic + (
            "\n\nCanonical Stage A JSON contract:\n"
            "- Output exactly one JSON object and no markdown, prose, bullets, or code fence.\n"
            "- The object must have exactly these top-level keys: action, evidence_status, tool_calls, cited_source_ids, rationale.\n"
            "- action must be one short enum token: ground, reject, defer, verify, flag, or self_answer.\n"
            "- Do not output long action names such as verify_with_assay_or_database; use verify.\n"
            "- evidence_status must be one short enum token: supported, contradicted, invalid_value, insufficient, or unknown.\n"
            "- evidence_status=verified is invalid. If a claim needs checking, use action=verify and evidence_status=insufficient.\n"
            "- evidence_status=valid, sourced, positive, negative, mixed, or related is invalid.\n"
            "- tool_calls must be a list of exactly four objects in this order:\n"
            "  1. {\"name\":\"nullatlas_survey_prior_failures\",\"arguments\":{\"drug_id\":\"<drug_id>\",\"condition_id\":\"<condition_id>\"}}\n"
            "  2. {\"name\":\"nullatlas_verify_trial_claims\",\"arguments\":{\"drug_id\":\"<drug_id>\",\"condition_id\":\"<condition_id>\"}}\n"
            "  3. {\"name\":\"nullatlas_check_value_validity\",\"arguments\":{\"drug_id\":\"<drug_id>\",\"condition_id\":\"<condition_id>\"}}\n"
            "  4. {\"name\":\"nullatlas_negative_evidence_completeness\",\"arguments\":{\"drug_id\":\"<drug_id>\",\"condition_id\":\"<condition_id>\"}}\n"
            "- Replace placeholder argument values only when exact drug_id and condition_id are visible in the user task.\n"
            "- cited_source_ids must be a JSON list. Include only source IDs visible in the prompt; otherwise use [].\n"
            "- Do not invent NCT IDs, trial IDs, hidden labels, gold status, expected action, case family, source task IDs, or evaluator metadata.\n"
            "- Example of a valid enum pair: {\"action\":\"verify\",\"evidence_status\":\"insufficient\"}."
        )
    raise ValueError(f"Unknown prompt_contract: {prompt_contract}")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=("oracle", "self_answer", "compact_tool_names_oracle", "openai_chat", "hf_chat"),
        default="self_answer",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model id. Defaults to gpt-4.1-mini for openai_chat and "
            "Qwen/Qwen2.5-0.5B-Instruct for hf_chat."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-contract", choices=PROMPT_CONTRACTS, default="basic")
    parser.add_argument(
        "--allow-live-api",
        action="store_true",
        help="Required for openai_chat mode; public smoke paths should omit this.",
    )
    parser.add_argument(
        "--allow-model-load",
        action="store_true",
        help="Required for hf_chat mode; intended for cluster/HPC jobs.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads for hf_chat mode. Defaults to existing HF cache only.",
    )
    args = parser.parse_args()
    model_id = effective_model(args.mode, args.model)

    rows = generate_rows(
        load_jsonl(args.sft),
        mode=args.mode,
        run_id=args.run_id,
        model=model_id,
        allow_live_api=args.allow_live_api,
        allow_model_load=args.allow_model_load,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        local_files_only=not args.allow_download,
        prompt_contract=args.prompt_contract,
        limit=args.limit,
    )
    write_jsonl(args.out, rows)
    print(
        json.dumps(
            {
                "dataset": DATASET,
                "mode": args.mode,
                "model": model_id,
                "out": args.out,
                "prompt_contract": args.prompt_contract,
                "rows": len(rows),
                "run_id": args.run_id,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
