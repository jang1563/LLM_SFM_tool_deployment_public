#!/usr/bin/env python3
"""Fuzzy-retrieval task set, built at the CHEMBL (drug) level — the correct granularity for
"Has drug X been tested and failed for indication Y?" (X = the active molecule = chembl_id, NOT a
single registry surface form).

Design: NEGBIODB_CT_FUZZY_RETRIEVAL_DESIGN_2026-06-26.md (+ chembl-gold correction 2026-06-26).
Each task: a (chembl_id C, condition Y) with a known chembl-level gold, queried by the most
string-DISSIMILAR surface form of C (the brand↔generic problem). The failure(s) live under sibling
forms -> exact/fuzzy-string resolvers miss them (false-deprioritization), the synonym(chembl)
resolver recovers them. condition_id stays oracle.

Gold (chembl level):
  ground : C has an efficacy failure for Y (any form), endpoints not mixed  -> cite an NCT
  flag   : ground-like, tagged for impossible-value injection
  reject : C has BOTH met and not-met endpoints for Y (mixed across forms)
  verify : C has failures for OTHER indications but none for Y
  defer  : C has NO recorded failures for ANY indication
"""
import argparse, json, re, sqlite3
from collections import Counter, defaultdict

try:
    from negbiodb_ct.paths import NEGBIODB_ROOT, CT_DB
except ImportError:
    from paths import NEGBIODB_ROOT, CT_DB

NEGBIODB = NEGBIODB_ROOT
ACTIONS = ["ground", "reject", "defer", "verify", "flag"]
try:
    from rapidfuzz import fuzz
    def sim(a, b): return fuzz.ratio(a or "", b or "") / 100.0
except ImportError:
    import difflib
    def sim(a, b): return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def extract_nct(s):
    m = re.search(r"(NCT\d+)", s or "")
    return m.group(1) if m else None


def render(drug, condition):
    return f"Has {drug} been tested and failed for {condition}? Use the available tools, then state your conclusion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "tasks_fuzzy.jsonl"))
    a = ap.parse_args()
    import random
    rng = random.Random(a.seed)
    con = sqlite3.connect(CT_DB); con.row_factory = sqlite3.Row

    cond_label = {int(r["condition_id"]): (r["condition_name"] or f"condition:{r['condition_id']}")
                  for r in con.execute("SELECT condition_id, condition_name FROM conditions")}
    all_conditions = list(cond_label)
    # chembl_id -> surface forms (id, name); only multi-form drugs (>=2) are queryable cross-form
    forms = defaultdict(list)
    for r in con.execute("SELECT intervention_id, intervention_name, chembl_id FROM interventions "
                        "WHERE intervention_type IN ('drug','biologic') AND chembl_id IS NOT NULL AND intervention_name IS NOT NULL"):
        forms[r["chembl_id"]].append((r["intervention_id"], r["intervention_name"]))
    multi = {c: f for c, f in forms.items() if len({n.lower() for _, n in f}) >= 2}

    # chembl x condition failure structure (aggregated across forms)
    cc = defaultdict(lambda: {"eff_nct": [], "met": set()})   # (chembl,cond) -> efficacy NCTs + endpoint set
    chembl_conds = defaultdict(set)                            # chembl -> conditions it has any failure for
    chembl_any = set()
    id2chembl = {r["intervention_id"]: r["chembl_id"] for r in con.execute(
        "SELECT intervention_id, chembl_id FROM interventions WHERE chembl_id IS NOT NULL")}
    for r in con.execute("SELECT intervention_id, condition_id, source_record_id, primary_endpoint_met, failure_category "
                       "FROM trial_failure_results WHERE source_record_id LIKE '%NCT%'"):
        ch = id2chembl.get(r["intervention_id"])
        if ch is None or ch not in multi:
            continue
        key = (ch, r["condition_id"])
        chembl_conds[ch].add(r["condition_id"]); chembl_any.add(ch)
        if r["failure_category"] == "efficacy":
            nct = extract_nct(r["source_record_id"])
            if nct:
                cc[key]["eff_nct"].append(nct)
        if r["primary_endpoint_met"] is not None:
            cc[key]["met"].add(r["primary_endpoint_met"])

    prop = {"ground": 0.35, "defer": 0.30, "verify": 0.15, "reject": 0.10, "flag": 0.10}
    want = {k: round(a.n * v) for k, v in prop.items()}
    tasks, used = [], set()

    def query_form(ch):                       # most string-dissimilar surface form -> hardest cross-form
        fs = multi[ch]
        canon = min((n for _, n in fs), key=len)   # shortest ~ the generic/canonical
        qid, qname = min(fs, key=lambda x: sim(x[1], canon))
        return qid, qname, round(sim(qname, canon), 3)

    def add(cls, ch, cond, gold, nct=None):
        if (ch, cond, cls) in used:
            return False
        qid, qname, s = query_form(ch)
        obs_claim = render(qname, cond_label.get(cond, f"condition:{cond}"))
        if nct and nct.lower() in obs_claim.lower():
            raise AssertionError(f"LEAK {nct}")
        used.add((ch, cond, cls))
        tasks.append({
            "packet_id": f"fz::{cls}::{ch}::{cond}", "action_class": cls, "available_actions": ACTIONS,
            "observation": {"claim": obs_claim, "query_name": qname, "condition_id": cond,
                            "drug_id": qid,   # the queried surface form (evidence-packet id; NOT used for retrieval — that goes via query_name)
                            "is_cross_form": s < 0.999, "name_similarity": s},
            "scoring_key": {"gold_action": gold, "gold_nct": nct, "inject_impossible_value": cls == "flag",
                            "chembl_id": ch, "query_id": qid},
        })
        return True

    keys = list(cc.items()); rng.shuffle(keys)
    # ground + flag: efficacy failure for Y, not mixed
    for (ch, cond), d in keys:
        if d["eff_nct"] and len(d["met"]) <= 1:
            if sum(t["action_class"] == "ground" for t in tasks) < want["ground"]:
                add("ground", ch, cond, "ground", d["eff_nct"][0])
            elif sum(t["action_class"] == "flag" for t in tasks) < want["flag"]:
                add("flag", ch, cond, "flag", d["eff_nct"][0])
    # reject: mixed endpoints for Y
    for (ch, cond), d in keys:
        if len(d["met"]) > 1 and sum(t["action_class"] == "reject" for t in tasks) < want["reject"]:
            add("reject", ch, cond, "reject")
    # verify: C has failures elsewhere, but a NOVEL condition for it
    cand = [c for c in multi if c in chembl_any]; rng.shuffle(cand)
    for ch in cand:
        if sum(t["action_class"] == "verify" for t in tasks) >= want["verify"]:
            break
        novel = [c for c in all_conditions if c not in chembl_conds[ch]]
        if novel:
            add("verify", ch, rng.choice(novel), "verify")
    # defer: C with NO failures anywhere
    nofail = [c for c in multi if c not in chembl_any]; rng.shuffle(nofail)
    for ch in nofail:
        if sum(t["action_class"] == "defer" for t in tasks) >= want["defer"]:
            break
        add("defer", ch, rng.choice(all_conditions), "defer")

    rng.shuffle(tasks)
    Path(a.out).write_text("".join(json.dumps(t) + "\n" for t in tasks))
    bal = Counter(t["action_class"] for t in tasks)
    sims = sorted(t["observation"]["name_similarity"] for t in tasks if t["observation"]["is_cross_form"])
    print(f"wrote {len(tasks)} chembl-level fuzzy tasks -> {a.out}")
    print(f"class balance: {dict(bal)} / want {want}")
    print(f"cross-form: {sum(t['observation']['is_cross_form'] for t in tasks)}/{len(tasks)}; "
          f"name-sim min {sims[0]:.2f}/median {sims[len(sims)//2]:.2f}/max {sims[-1]:.2f}" if sims else "")
    leaks = sum(1 for t in tasks if t["scoring_key"]["gold_nct"]
                and t["scoring_key"]["gold_nct"].lower() in t["observation"]["claim"].lower())
    print(f"leak-audit: {leaks} (must be 0)"); assert leaks == 0


if __name__ == "__main__":
    main()
