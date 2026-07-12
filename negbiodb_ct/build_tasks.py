#!/usr/bin/env python3
"""Build the balanced-action NegBioDB-CT trajectory task set (Gate-1 pilot).

Design: NEGBIODB_CT_TRAJECTORY_DESIGN_2026-06-25.md (sections 4-6).
Leak-safe: the rendered `observation` never contains the gold action, the NCT, the
failure_category, or DB-membership; truth lives only in `scoring_key`.
Non-circular: defer/verify use absent (drug x condition) combos the tool cannot relay;
ground uses a held-out split so it measures the LEARNED policy's generalization.

Action classes (gold):
  ground : recorded failure exists (held-out)                         -> cite NCT + category
  reject : same drug x indication has BOTH met and not-met endpoints  -> reject unqualified "failed"
  defer  : drug has ZERO recorded failures                            -> no evidence -> defer
  verify : drug HAS failures (other indications) x NEW indication      -> related evidence -> verify
  flag   : ground-like, tagged for impossible-value injection (adapter) -> flag
"""
import argparse, csv, json, os, random, re, sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEGBIODB = Path(os.environ.get("NEGBIODB_ROOT", ROOT.parent / "Negative_result_DB")).expanduser()
CT_DB = Path(os.environ.get("NEGBIODB_CT_DB", NEGBIODB / "data/negbiodb_ct.db")).expanduser()
SPLITS = Path(os.environ.get("NEGBIODB_CT_SPLITS", NEGBIODB / "exports/ct/negbiodb_ct_splits.csv")).expanduser()
ACTIONS = ["ground", "reject", "defer", "verify", "flag"]
DRUG_TYPES = ("drug", "biologic")


def extract_nct(s):
    m = re.search(r"(NCT\d+)", s or "")
    return m.group(1) if m else None


def drug_label(row):
    return (row["canonical_name"] or row["intervention_name"] or f"drug:{row['intervention_id']}")


def condition_labeler(con):
    m = {int(cid): nm for cid, nm in con.execute(
        "SELECT condition_id, condition_name FROM conditions WHERE condition_name IS NOT NULL")}
    print(f"[condition labels] conditions.condition_name ({len(m)})")
    return lambda cid: m.get(int(cid), f"condition:{cid}")


def render(drug, condition):  # NO truth, NO db-membership, NO NCT, NO category, NO action-name (neutral)
    return f"Has {drug} been tested and failed for {condition}? Use the available tools, then state your conclusion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--split", default="split_random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "tasks_pilot.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    con = sqlite3.connect(CT_DB); con.row_factory = sqlite3.Row
    cond_label = condition_labeler(con)

    prop = {"ground": 0.35, "defer": 0.30, "verify": 0.15, "reject": 0.10, "flag": 0.10}
    want = {k: round(a.n * v) for k, v in prop.items()}

    splits = list(csv.DictReader(open(SPLITS)))
    held = [(int(r["intervention_id"]), int(r["condition_id"])) for r in splits if r.get(a.split) == "test"]
    rng.shuffle(held)
    all_conditions = list({int(r["condition_id"]) for r in splits})

    # drug pool (drug/biologic only) + their recorded conditions
    # chembl_id IS NOT NULL keeps real, single-compound drugs -> matches the drug-name surface
    # distribution ACROSS classes (else defer's messy arm-descriptions leak the action; G2 confound).
    drug_info = {r["intervention_id"]: r for r in con.execute(
        f"SELECT intervention_id, intervention_name, canonical_name FROM interventions "
        f"WHERE intervention_type IN {DRUG_TYPES} AND chembl_id IS NOT NULL")}
    drug_conds = {}
    for iid, cid in con.execute("SELECT intervention_id, condition_id FROM trial_failure_results"):
        if iid in drug_info:
            drug_conds.setdefault(iid, set()).add(cid)
    drugs_with_fail = set(drug_conds)
    drugs_no_fail = [d for d in drug_info if d not in drugs_with_fail]
    rng.shuffle(drugs_no_fail)

    tasks, used = [], set()

    def add(cls, drug_id, condition_id, gold_action, nct=None, category=None, note=None):
        d = drug_info.get(drug_id)
        if not d or (drug_id, condition_id, cls) in used:
            return False
        obs = render(drug_label(d), cond_label(condition_id))
        if nct and nct.lower() in obs.lower():  # leak-guard: the NCT (the only real leak risk) must never appear
            raise AssertionError(f"LEAK '{nct}' in observation ({cls})")
        # category & gold_action are class labels, never rendered by construction -> no substring guard (false positives)
        used.add((drug_id, condition_id, cls))
        tasks.append({
            "packet_id": f"ct::{cls}::{drug_id}::{condition_id}",
            "action_class": cls, "available_actions": ACTIONS,
            "observation": {"claim": obs, "drug_id": drug_id, "condition_id": condition_id},
            "scoring_key": {"gold_action": gold_action, "gold_nct": nct,
                            "gold_failure_category": category,
                            "inject_impossible_value": cls == "flag", "note": note},
        })
        return True

    def ground_rec(iid, cid):
        # ground/flag require an EFFICACY failure: "did X FAIL for Y?" means the drug didn't work,
        # NOT a logistical termination (enrollment/strategic). The frontier reference (Sonnet) revealed
        # the looser gold: it correctly defers on non-efficacy terminations.
        mixed = con.execute("SELECT COUNT(DISTINCT primary_endpoint_met) FROM trial_failure_results "
                            "WHERE intervention_id=? AND condition_id=? "
                            "AND primary_endpoint_met IS NOT NULL", (iid, cid)).fetchone()[0]
        if mixed > 1:
            return (None, None)
        r = con.execute("SELECT source_record_id, failure_category FROM trial_failure_results "
                        "WHERE intervention_id=? AND condition_id=? AND failure_category='efficacy' "
                        "AND source_record_id LIKE '%NCT%' LIMIT 1",
                        (iid, cid)).fetchone()
        return (extract_nct(r[0]), r[1]) if r else (None, None)

    # ground + flag from held-out pairs with a real NCT failure record
    for iid, cid in held:
        if sum(t["action_class"] == "ground" for t in tasks) >= want["ground"]:
            break
        nct, cat = ground_rec(iid, cid)
        if nct:
            add("ground", iid, cid, "ground", nct, cat)
    for iid, cid in held:
        if sum(t["action_class"] == "flag" for t in tasks) >= want["flag"]:
            break
        nct, cat = ground_rec(iid, cid)
        if nct:
            add("flag", iid, cid, "flag", nct, cat,
                note="adapter injects an impossible p-value into the gold efficacy record")

    # reject: same drug x indication with BOTH met and not-met endpoints (mixed)
    rej = con.execute("SELECT intervention_id, condition_id FROM trial_failure_results "
                      "WHERE primary_endpoint_met IS NOT NULL GROUP BY intervention_id, condition_id "
                      "HAVING COUNT(DISTINCT primary_endpoint_met) > 1").fetchall()
    rng.shuffle(rej)
    for iid, cid in rej:
        if sum(t["action_class"] == "reject" for t in tasks) >= want["reject"]:
            break
        add("reject", iid, cid, "reject", note="both met and not-met endpoints for this drug x indication -> mixed, reject unqualified 'failed'")

    # defer: drug with zero recorded failures x a real condition
    for did in drugs_no_fail:
        if sum(t["action_class"] == "defer" for t in tasks) >= want["defer"]:
            break
        add("defer", did, rng.choice(all_conditions), "defer", note="drug has no recorded failures -> insufficient evidence")

    # verify: drug WITH failures (other indications) x a NEW indication for it
    cand = list(drugs_with_fail); rng.shuffle(cand)
    for did in cand:
        if sum(t["action_class"] == "verify" for t in tasks) >= want["verify"]:
            break
        novel = [c for c in all_conditions if c not in drug_conds[did]]
        if novel:
            add("verify", did, rng.choice(novel), "verify", note="related evidence (other indications) but not this one -> verify before asserting")

    rng.shuffle(tasks)
    Path(a.out).write_text("".join(json.dumps(t) + "\n" for t in tasks))

    bal = Counter(t["action_class"] for t in tasks)
    print(f"\nwrote {len(tasks)} tasks -> {a.out}")
    print("class balance:", dict(bal), "/ want", want)
    leaks = sum(1 for t in tasks if t["scoring_key"]["gold_nct"]
                and t["scoring_key"]["gold_nct"].lower() in t["observation"]["claim"].lower())
    print("leak-audit (NCT in claim):", leaks, "(must be 0)"); assert leaks == 0
    labelled = sum(1 for t in tasks if not t["observation"]["claim"].count("condition:"))
    print(f"readable-condition rate: {labelled}/{len(tasks)}")
    print(f"max single-class share: {bal.most_common(1)[0][1] / len(tasks):.0%} (constant-policy ceiling)")


if __name__ == "__main__":
    main()
