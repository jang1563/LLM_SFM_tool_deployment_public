#!/usr/bin/env python3
"""§13.3 — the trajectory agent runner (LLM x NullAtlas-CT tools), Gate-1 frontier reference.

Reuses the report's E2 pattern (scripts_rl/168): LLMClient + a JSON function-calling loop.
The model orchestrates NullAtlas-style tools, then emits a terminal action; we score vs held-out gold.
Pilot simplification (noted): tools execute against the task's KNOWN ids (oracle retrieval) -> this
isolates DECISION quality from name-resolution noise. The full experiment adds retrieval noise.
"""
import argparse, json, os, random, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEGBIODB = Path(os.environ.get("NEGBIODB_ROOT", ROOT.parent / "Negative_result_DB")).expanduser()
sys.path.insert(0, str(NEGBIODB / "src"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))
from baselines import score_decision, extract_nct  # reuse the scorer
from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import task_spec_from_record, trajectory_from_model_output

CT_DB = Path(os.environ.get("NEGBIODB_CT_DB", NEGBIODB / "data/negbiodb_ct.db")).expanduser()

SYS = (
    "You decide whether a drug has been tested and FAILED for an indication, using tools. "
    "Respond with JSON only, one object per turn.\n"
    "Tools:\n"
    '  {"tool":"search_failures"} -> recorded failed trials for THIS drug+indication '
    "(list of {nct, endpoint_met (0=not met,1=met), p_value, year})\n"
    '  {"tool":"check_other_indications"} -> count of the drug\'s recorded failures for OTHER indications\n'
    "Then give your FINAL decision (JSON), exactly one of:\n"
    '  {"action":"ground","nct":"NCT........"}  - a clean recorded failure exists; cite ONE returned NCT\n'
    '  {"action":"reject"}  - mixed outcomes (some trials met the endpoint, some did not): unqualified "failed" is wrong\n'
    '  {"action":"flag","nct":"NCT........"}    - a returned record has a physically impossible value; cite that NCT\n'
    '  {"action":"verify"}  - no record for THIS indication, but the drug failed for OTHER indications: verify first\n'
    '  {"action":"defer"}   - no recorded evidence at all: insufficient to assert\n'
    "Call the tools before deciding. Cite an NCT only for ground or flag, and only one the tool returned. "
    "If returned rows contain both endpoint_met=0 and endpoint_met=1, choose reject; this overrides ground."
)

SYS_NATIVE = (
    "You decide whether a drug has been tested and FAILED for an indication. "
    "First call the evidence tools, THEN call submit_decision exactly once.\n"
    "'Failed' means an EFFICACY failure (the drug did not work) -- a non-efficacy termination "
    "(enrollment/strategic/other) is NOT the drug failing.\n"
    "If returned rows contain BOTH endpoint_met=0 and endpoint_met=1 for this drug+indication, "
    "choose reject; mixed endpoint evidence overrides grounding from any single efficacy row.\n"
    "Decide one of: ground (a recorded efficacy failure exists -- pass its nct); "
    "reject (mixed: some trials met the primary endpoint, some did not); "
    "flag (a returned record has a physically impossible value, e.g. p_value outside [0,1] -- pass its nct); "
    "verify (no efficacy failure for THIS indication, but the drug failed for OTHER indications); "
    "defer (no recorded efficacy failure for this indication -- no record, or only non-efficacy terminations)."
)

TOOLS = [
    {"name": "search_failures",
     "description": "Recorded failed trials for THIS drug+indication: list of {nct, endpoint_met (0=not met,1=met), p_value, year, failure_category}. If both endpoint_met=0 and endpoint_met=1 appear, reject the unqualified failed claim. Otherwise, failure_category='efficacy' is the decisive failure label, even when endpoint_met or p_value is null.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "check_other_indications",
     "description": "How many recorded failures the drug has for OTHER indications.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "submit_decision",
     "description": "Submit your final decision (call exactly once, after using the evidence tools).",
     "input_schema": {"type": "object",
                      "properties": {
                          "action": {"type": "string", "enum": ["ground", "reject", "defer", "verify", "flag"]},
                          "nct": {"type": "string", "description": "the cited NCT id (ground or flag only)"}},
                      "required": ["action"]}},
]


def search_failures(con, did, cid, inject_nct=None):
    rows = con.execute(
        "SELECT source_record_id, primary_endpoint_met, p_value_primary, publication_year, failure_category "
        "FROM trial_failure_results WHERE intervention_id=? AND condition_id=? AND source_record_id LIKE '%NCT%'",
        (did, cid)).fetchall()
    out = []
    for r in rows:
        nct = extract_nct(r[0])
        pval = 14.73 if (inject_nct and nct == inject_nct) else r[2]
        out.append({"nct": nct, "endpoint_met": r[1], "p_value": pval,
                    "year": r[3], "failure_category": r[4]})
    return out


def check_other(con, did, cid):
    n = con.execute("SELECT COUNT(*) FROM trial_failure_results WHERE intervention_id=? AND condition_id!=?",
                    (did, cid)).fetchone()[0]
    return {"failures_for_other_indications": n}


def parse_json(s):
    m = re.search(r"\{.*\}", s or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def run_task(aclient, model, con, t, max_steps=6):
    """Anthropic native tool-use loop: clean tool_use blocks + a submit_decision tool force the format
    and stop the model from simulating tool results (the 3 JSON-protocol bugs)."""
    did, cid = t["observation"]["drug_id"], t["observation"]["condition_id"]
    inject_nct = t["scoring_key"]["gold_nct"] if t["scoring_key"]["inject_impossible_value"] else None
    messages = [{"role": "user", "content": t["observation"]["claim"]}]
    called = []
    for _ in range(max_steps):
        resp = aclient.messages.create(model=model, system=SYS_NATIVE, tools=TOOLS,
                                       max_tokens=1024, temperature=0.0, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            messages.append({"role": "user", "content": "Call submit_decision now with your final action."})
            continue
        results, decision = [], None
        for b in resp.content:
            if b.type != "tool_use":
                continue
            if b.name == "submit_decision":
                decision = b.input
            elif b.name == "search_failures":
                called.append(b.name)
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(search_failures(con, did, cid, inject_nct))})
            elif b.name == "check_other_indications":
                called.append(b.name)
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(check_other(con, did, cid))})
        if decision is not None:
            act = decision.get("action")
            return {"action": act, "cited_nct": extract_nct(decision.get("nct", "")) if act in {"ground", "flag"} else None}, called
        messages.append({"role": "user", "content": results})
    return {"action": None, "cited_nct": None}, called


def generic_model_output(decision, called):
    """Convert the pilot runner's compact decision into parser-compatible JSON."""

    action = decision.get("action") or "self_answer"
    out = {"action": action, "called": called}
    if decision.get("cited_nct"):
        out["cited_source_ids"] = [decision["cited_nct"]]
    return out


def generic_score(task, decision, called):
    """Score the runner output with the shared trajectory evaluator."""

    output = generic_model_output(decision, called)
    trajectory = trajectory_from_model_output(task, output, tool_profile="native_ct")
    result = TrajectoryEvaluator().evaluate(
        task_spec_from_record(task, tool_profile="native_ct"),
        trajectory,
    )
    return output, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_pilot.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "agent_sonnet.json"))
    ap.add_argument("--packet-id", action="append", default=[],
                    help="Run exact packet_id(s), bypassing class-balanced sampling. Repeatable.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    aclient = None
    if not a.dry_run:
        import anthropic
        from negbiodb.llm_client import LLMClient
        aclient = anthropic.Anthropic(api_key=LLMClient(provider="anthropic", model=a.model).api_key)  # reuse key resolution
    con = sqlite3.connect(CT_DB)
    tasks = [json.loads(l) for l in open(a.tasks)]
    if a.packet_id:
        wanted = set(a.packet_id)
        sample = [t for t in tasks if t["packet_id"] in wanted]
        missing = wanted - {t["packet_id"] for t in sample}
        if missing:
            raise SystemExit(f"packet_id(s) not found: {sorted(missing)}")
    else:
        rng = random.Random(42)
        by = defaultdict(list)
        for t in tasks:
            by[t["action_class"]].append(t)
        per = max(1, a.n // len(by))
        sample = []
        for c, ts in by.items():
            sample += rng.sample(ts, min(per, len(ts)))
        rng.shuffle(sample)

    rows, bycls, rew, toolrate, generic_scores = [], defaultdict(lambda: [0, 0]), [], 0, []
    for i, t in enumerate(sample):
        if a.dry_run:
            dec, called = {"action": None, "cited_nct": None}, []
        else:
            dec, called = run_task(aclient, a.model, con, t)
        s = score_decision(dec, t["scoring_key"])
        parsed_output, generic_result = generic_score(t, dec, called)
        rew.append(s["reward"]); bycls[t["action_class"]][0] += s["correct"]; bycls[t["action_class"]][1] += 1
        generic_scores.append(generic_result.score)
        toolrate += bool(called)
        rows.append({"packet_id": t["packet_id"],
                     "class": t["action_class"], "gold": t["scoring_key"]["gold_action"],
                     "pred": dec, "called": called, "model_output": parsed_output,
                     "correct": s["correct"], "reward": s["reward"],
                     "generic_score": round(generic_result.score, 3),
                     "generic_violations": list(generic_result.violations)})
        print(f"  [{i:2d}] {t['action_class']:7s} pred={str(dec['action']):7s} gold={t['scoring_key']['gold_action']:7s} "
              f"{'OK' if s['correct'] else 'X '} tools={called} generic={generic_result.score:.2f}",
              flush=True)
    summary = {"model": a.model, "n": len(sample),
               "action_accuracy": round(sum(c for c, _ in bycls.values()) / len(sample), 3),
               "mean_reward": round(sum(rew) / len(rew), 3),
               "generic_mean_score": round(sum(generic_scores) / len(generic_scores), 3),
               "tool_call_rate": round(toolrate / len(sample), 3),
               "by_class": {c: f"{v[0]}/{v[1]}" for c, v in sorted(bycls.items())}}
    print("\n=== Sonnet frontier reference (Gate-1) ===\n" + json.dumps(summary, indent=2))
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
