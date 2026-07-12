#!/usr/bin/env python3
"""§13.2 — scorer + no-API baselines for the NegBioDB-CT trajectory tasks.

Runs locally (direct DB queries; no API/HPC). Produces:
  - G2 confound check: does a claim-surface rule (no tools) predict the gold action above chance?
  - mechanics ceiling / sanity: does a deterministic tool-policy (DB lookup) solve the tasks?
The LLM conditions (prompting-only open / frontier) reuse this scorer but need API/HPC -> later.
"""
import argparse, json, re, sqlite3
from collections import Counter, defaultdict

try:
    from negbiodb_ct.paths import NEGBIODB_ROOT, CT_DB
except ImportError:
    from paths import NEGBIODB_ROOT, CT_DB

NEGBIODB = NEGBIODB_ROOT


def extract_nct(s):
    m = re.search(r"(NCT\d+)", s or "")
    return m.group(1) if m else None


def score_decision(decision, gold):
    """decision = {action, cited_nct}; gold = scoring_key. Encodes the report's lessons:
    misattribution (ground+wrong NCT) = fail; fabrication (ground when gold!=ground) = penalized."""
    a, ga = decision.get("action"), gold["gold_action"]
    if ga == "ground":
        ok = (a == "ground" and decision.get("cited_nct") == gold["gold_nct"])
        r = 1.0 if ok else (0.0 if a == "ground" else 0.0)  # wrong NCT = misattribution = 0
        return {"correct": a == ga, "attribution_ok": (a == "ground" and decision.get("cited_nct") == gold["gold_nct"]), "reward": r}
    # gold != ground
    if a == "ground":
        return {"correct": False, "attribution_ok": None, "reward": -0.5}  # fabricated a citation = false graveyard
    return {"correct": a == ga, "attribution_ok": None, "reward": 1.0 if a == ga else 0.0}


def deterministic_policy(task, con):
    """Ideal tool-use: query the DB (= perfect tool calls) and route. Re-derives the action from the
    same evidence the builder used -> the mechanics CEILING + a solvability sanity check."""
    iid, cid = task["observation"]["drug_id"], task["observation"]["condition_id"]
    inject = task["scoring_key"]["inject_impossible_value"]
    efficacy_recs = con.execute("SELECT source_record_id FROM trial_failure_results "
                                "WHERE intervention_id=? AND condition_id=? "
                                "AND failure_category='efficacy' AND source_record_id LIKE '%NCT%'",
                                (iid, cid)).fetchall()
    if efficacy_recs and inject:              # check_value_validity fails -> flag
        return {"action": "flag", "cited_nct": None}
    met = {m[0] for m in con.execute(
        "SELECT DISTINCT primary_endpoint_met FROM trial_failure_results "
        "WHERE intervention_id=? AND condition_id=? AND primary_endpoint_met IS NOT NULL", (iid, cid))}
    if len(met) > 1:                          # mixed endpoints -> reject
        return {"action": "reject", "cited_nct": None}
    if efficacy_recs:
        return {"action": "ground", "cited_nct": extract_nct(efficacy_recs[0][0])}
    other = con.execute("SELECT 1 FROM trial_failure_results WHERE intervention_id=? LIMIT 1", (iid,)).fetchone()
    return {"action": "verify" if other else "defer", "cited_nct": None}  # related evidence -> verify; none -> defer


def surface_rules(tasks):
    """G2: best claim-SURFACE rules (no tools). If any beats the majority ceiling, the task leaks
    the answer through the claim surface (a confound)."""
    golds = [t["scoring_key"]["gold_action"] for t in tasks]
    maj = Counter(golds).most_common(1)[0]
    out = {"majority_class": (maj[0], maj[1] / len(tasks))}
    # candidate surface features (claim text only)
    feats = {
        "drug_name_len": lambda t: len(t["observation"]["claim"].split(" been tested")[0]),
        "claim_len": lambda t: len(t["observation"]["claim"]),
        "drug_word_count": lambda t: len(t["observation"]["claim"].split(" been tested")[0].split()),
    }
    best = {}
    for fn_name, fn in feats.items():
        xs = sorted({fn(t) for t in tasks})
        bestacc = 0.0
        for thr in xs:                        # best single threshold -> predict majority-above vs -below
            for hi, lo in [(maj[0], None)]:   # threshold split, assign each side its own majority
                left = [t["scoring_key"]["gold_action"] for t in tasks if fn(t) <= thr]
                right = [t["scoring_key"]["gold_action"] for t in tasks if fn(t) > thr]
                if not left or not right:
                    continue
                lm = Counter(left).most_common(1)[0][0]
                rm = Counter(right).most_common(1)[0][0]
                acc = sum((fn(t) <= thr and t["scoring_key"]["gold_action"] == lm) or
                          (fn(t) > thr and t["scoring_key"]["gold_action"] == rm) for t in tasks) / len(tasks)
                bestacc = max(bestacc, acc)
        best[fn_name] = round(bestacc, 3)
    out["best_surface_threshold_acc"] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_pilot.jsonl"))
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks)]
    con = sqlite3.connect(CT_DB)

    # deterministic tool-policy
    by_class = defaultdict(lambda: [0, 0])
    rewards = []
    for t in tasks:
        d = deterministic_policy(t, con)
        s = score_decision(d, t["scoring_key"])
        rewards.append(s["reward"])
        by_class[t["action_class"]][0] += s["correct"]
        by_class[t["action_class"]][1] += 1
    print("=== deterministic tool-policy (mechanics ceiling / solvability sanity) ===")
    print(f"  overall action-accuracy: {sum(s for s,_ in by_class.values())/len(tasks):.3f}"
          if False else f"  overall action-accuracy: {sum(c for c,_ in by_class.values())/len(tasks):.3f}")
    print(f"  mean reward: {sum(rewards)/len(rewards):+.3f}")
    for c, (corr, n) in sorted(by_class.items()):
        print(f"    {c:7s}: {corr}/{n} = {corr/n:.2f}")

    print("\n=== G2 confound check: claim-surface rules (no tools) ===")
    g2 = surface_rules(tasks)
    print(f"  majority-class ceiling: {g2['majority_class'][0]} = {g2['majority_class'][1]:.3f}")
    print(f"  best single-feature threshold accuracy: {g2['best_surface_threshold_acc']}")
    worst = max(g2['best_surface_threshold_acc'].values())
    verdict = "PASS (no surface confound)" if worst <= g2['majority_class'][1] + 0.10 else \
              "WARN: a claim-surface feature predicts the action -> possible confound"
    print(f"  G2 verdict: {verdict}  (best surface {worst:.3f} vs majority {g2['majority_class'][1]:.3f})")


if __name__ == "__main__":
    main()
