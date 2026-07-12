#!/usr/bin/env python3
"""Fuzzy-retrieval runner — same model interface as run_agent_open.py, but the drug tools are backed
by a name RESOLVER (tier = exact | fuzzy | synonym) instead of oracle ids. condition_id stays oracle.

The model is BLIND to the tier (it just calls search_failures / check_other_indications); the tier is
the experimental condition. Compares: does a weak retriever make the model false-deprioritize, and does
the synonym tier recover it? Reuses the OpenAI tool-use loop + the scorer + the shared evaluator.

Usage (HPC, vLLM serving the model with --enable-auto-tool-choice):
  python run_agent_open_fuzzy.py --model <served> --base-url http://localhost:8000/v1 --tier exact --n 400
"""
import argparse, json, random, sqlite3, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from negbiodb_ct.paths import CT_DB
from baselines import score_decision
from fuzzy_resolvers import Resolver, failures_for, others_count, extract_nct
from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import task_spec_from_record, trajectory_from_model_output

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

OPENAI_TOOLS = [
    {"type": "function", "function": {
        "name": "search_failures",
        "description": "Recorded failed trials for THIS drug+indication: list of {nct, endpoint_met (0=not met,1=met), p_value, year, failure_category}. If both endpoint_met=0 and endpoint_met=1 appear, reject the unqualified failed claim. Otherwise, failure_category='efficacy' is the decisive failure label, even when endpoint_met or p_value is null.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_other_indications",
        "description": "How many recorded failures the drug has for OTHER indications.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "submit_decision",
        "description": "Submit your final decision (call exactly once, after using the evidence tools).",
        "parameters": {"type": "object",
                       "properties": {"action": {"type": "string", "enum": ["ground", "reject", "defer", "verify", "flag"]},
                                      "nct": {"type": "string", "description": "the cited NCT id (ground or flag only)"}},
                       "required": ["action"]}}},
]


def run_task(client, model, con, R, tier, t, max_steps=6):
    o, sk = t["observation"], t["scoring_key"]
    ids = R.resolve(tier, o["query_name"])                       # NAME -> ids via the tier (condition stays oracle)
    cond = o["condition_id"]
    inject = sk["gold_nct"] if sk["inject_impossible_value"] else None
    messages = [{"role": "system", "content": SYS_NATIVE}, {"role": "user", "content": o["claim"]}]
    called = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=messages, tools=OPENAI_TOOLS,
                                              temperature=0.0, max_tokens=1024)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": "Call submit_decision now with your final action."})
            continue
        messages.append(msg.model_dump())
        decision = None
        for tc in msg.tool_calls:
            name = tc.function.name
            if name == "submit_decision":
                try:
                    decision = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    decision = {}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
            elif name == "search_failures":
                called.append(name)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(failures_for(con, ids, cond, inject))})
            elif name == "check_other_indications":
                called.append(name)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"failures_for_other_indications": others_count(con, ids, cond)})})
            else:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "unknown tool"})
        if decision is not None:
            act = decision.get("action")
            return {"action": act, "cited_nct": extract_nct(decision.get("nct", "")) if act in {"ground", "flag"} else None}, called
    return {"action": None, "cited_nct": None}, called


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tier", required=True, choices=["exact", "fuzzy", "synonym"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_fuzzy.jsonl"))
    ap.add_argument("--ct-db", default=str(CT_DB))
    ap.add_argument("--out", default=str(Path(__file__).parent / "fuzzy_open.json"))
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="EMPTY")
    con = sqlite3.connect(a.ct_db)
    R = Resolver(con)
    tasks = [json.loads(l) for l in open(a.tasks)]
    rng = random.Random(42)
    by = defaultdict(list)
    for t in tasks:
        by[t["action_class"]].append(t)
    per = max(1, a.n // len(by))
    sample = []
    for c, ts in by.items():
        sample += rng.sample(ts, min(per, len(ts)))
    rng.shuffle(sample)

    ev = TrajectoryEvaluator()
    rows, bycls, rew, toolrate, gen, fdep = [], defaultdict(lambda: [0, 0]), [], 0, [], [0, 0]
    for i, t in enumerate(sample):
        dec, called = run_task(client, a.model, con, R, a.tier, t)
        s = score_decision(dec, t["scoring_key"])
        out = {"action": dec.get("action") or "self_answer", "called": called}
        if dec.get("cited_nct"):
            out["cited_source_ids"] = [dec["cited_nct"]]
        g = ev.evaluate(task_spec_from_record(t, tool_profile="native_ct"),
                        trajectory_from_model_output(t, out, tool_profile="native_ct"))
        rew.append(s["reward"]); bycls[t["action_class"]][0] += s["correct"]; bycls[t["action_class"]][1] += 1
        gen.append(g.score); toolrate += bool(called)
        # false-deprioritization: gold says a failure exists, model said none (defer/verify)
        if t["scoring_key"]["gold_action"] in ("ground", "flag") and dec.get("action") in ("defer", "verify"):
            fdep[0] += 1
        if t["scoring_key"]["gold_action"] in ("ground", "flag"):
            fdep[1] += 1
        rows.append({"packet_id": t["packet_id"], "class": t["action_class"], "gold": t["scoring_key"]["gold_action"],
                     "pred": dec, "correct": s["correct"], "is_cross_form": t["observation"]["is_cross_form"]})
        print(f"  [{i:3d}] {t['action_class']:7s} pred={str(dec['action']):7s} gold={t['scoring_key']['gold_action']:7s} "
              f"{'OK' if s['correct'] else 'X '}", flush=True)
    summary = {"model": a.model, "tier": a.tier, "n": len(sample),
               "action_accuracy": round(sum(c for c, _ in bycls.values()) / len(sample), 3),
               "mean_reward": round(sum(rew) / len(rew), 3),
               "generic_mean_score": round(sum(gen) / len(gen), 3),
               "tool_call_rate": round(toolrate / len(sample), 3),
               "false_deprioritization_rate": round(fdep[0] / fdep[1], 3) if fdep[1] else None,
               "by_class": {c: f"{v[0]}/{v[1]}" for c, v in sorted(bycls.items())}}
    print(f"\n=== fuzzy {a.model} tier={a.tier} ===\n" + json.dumps(summary, indent=2))
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
