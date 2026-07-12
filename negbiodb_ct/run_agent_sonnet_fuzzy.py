#!/usr/bin/env python3
"""Fuzzy-retrieval FRONTIER reference (Sonnet, Anthropic-native tool-use) — the same resolver-backed
tools as run_agent_open_fuzzy.py, for the H2 comparison (does even a frontier model false-deprioritize
under a weak retriever?). Separate file so it never clobbers run_agent.py.

  python run_agent_sonnet_fuzzy.py --tier exact --n 100 --out fuzzy_sonnet_exact.json
"""
import argparse, json, random, sqlite3, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from negbiodb_ct.paths import NEGBIODB_ROOT
NEGBIODB = NEGBIODB_ROOT
sys.path.insert(0, str(NEGBIODB / "src")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
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
                      "properties": {"action": {"type": "string", "enum": ["ground", "reject", "defer", "verify", "flag"]},
                                     "nct": {"type": "string", "description": "the cited NCT id (ground or flag only)"}},
                      "required": ["action"]}},
]


def run_task(aclient, model, con, R, tier, t, max_steps=6):
    o, sk = t["observation"], t["scoring_key"]
    ids = R.resolve(tier, o["query_name"]); cond = o["condition_id"]
    inject = sk["gold_nct"] if sk["inject_impossible_value"] else None
    messages = [{"role": "user", "content": o["claim"]}]
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
                                "content": json.dumps(failures_for(con, ids, cond, inject))})
            elif b.name == "check_other_indications":
                called.append(b.name)
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps({"failures_for_other_indications": others_count(con, ids, cond)})})
        if decision is not None:
            act = decision.get("action")
            return {"action": act, "cited_nct": extract_nct(decision.get("nct", "")) if act in {"ground", "flag"} else None}, called
        messages.append({"role": "user", "content": results})
    return {"action": None, "cited_nct": None}, called


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--tier", required=True, choices=["exact", "fuzzy", "synonym"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_fuzzy.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "fuzzy_sonnet.json"))
    a = ap.parse_args()
    import anthropic
    from negbiodb.llm_client import LLMClient
    aclient = anthropic.Anthropic(api_key=LLMClient(provider="anthropic", model=a.model).api_key)
    con = sqlite3.connect(NEGBIODB / "data/negbiodb_ct.db")
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
    rows, bycls, rew, gen, fdep = [], defaultdict(lambda: [0, 0]), [], [], [0, 0]
    for i, t in enumerate(sample):
        dec, called = run_task(aclient, a.model, con, R, a.tier, t)
        s = score_decision(dec, t["scoring_key"])
        out = {"action": dec.get("action") or "self_answer", "called": called}
        if dec.get("cited_nct"):
            out["cited_source_ids"] = [dec["cited_nct"]]
        g = ev.evaluate(task_spec_from_record(t, tool_profile="native_ct"),
                        trajectory_from_model_output(t, out, tool_profile="native_ct"))
        rew.append(s["reward"]); bycls[t["action_class"]][0] += s["correct"]; bycls[t["action_class"]][1] += 1
        gen.append(g.score)
        if t["scoring_key"]["gold_action"] in ("ground", "flag"):
            fdep[1] += 1
            if dec.get("action") in ("defer", "verify"):
                fdep[0] += 1
        rows.append({"packet_id": t["packet_id"], "class": t["action_class"], "gold": t["scoring_key"]["gold_action"],
                     "pred": dec, "correct": s["correct"], "is_cross_form": t["observation"]["is_cross_form"]})
        print(f"  [{i:3d}] {t['action_class']:7s} pred={str(dec['action']):7s} gold={t['scoring_key']['gold_action']:7s} "
              f"{'OK' if s['correct'] else 'X '}", flush=True)
    summary = {"model": a.model, "tier": a.tier, "n": len(sample),
               "action_accuracy": round(sum(c for c, _ in bycls.values()) / len(sample), 3),
               "mean_reward": round(sum(rew) / len(rew), 3),
               "generic_mean_score": round(sum(gen) / len(gen), 3),
               "false_deprioritization_rate": round(fdep[0] / fdep[1], 3) if fdep[1] else None,
               "by_class": {c: f"{v[0]}/{v[1]}" for c, v in sorted(bycls.items())}}
    print(f"\n=== fuzzy SONNET tier={a.tier} ===\n" + json.dumps(summary, indent=2))
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
