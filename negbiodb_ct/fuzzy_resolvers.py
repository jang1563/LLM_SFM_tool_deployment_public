#!/usr/bin/env python3
"""3-tier drug-name resolver + the deterministic experiment-validity gate for the fuzzy-retrieval study.

Tiers (condition_id held oracle; only the drug NAME is resolved):
  exact   : lower(name)==lower(intervention_name)
  fuzzy   : rapidfuzz top-k over intervention_name (catches typos/dose; FAILS brand->generic)
  synonym : fuzzy-find the query id(s) -> their chembl_id(s) -> ALL forms of those chembl_ids
Run with no args -> the deterministic per-tier ceiling + cross-form ground-recall. MUST show
exact/fuzzy collapse on cross-form ground, synonym recover -> the experiment is well-posed.
"""
import argparse, json, re, sqlite3
from collections import Counter, defaultdict

try:
    from negbiodb_ct.paths import NEGBIODB_ROOT, CT_DB
except ImportError:
    from paths import NEGBIODB_ROOT, CT_DB

NEGBIODB = NEGBIODB_ROOT
try:
    from rapidfuzz import process, fuzz
    HAVE_RF = True
except ImportError:
    HAVE_RF = False


def extract_nct(s):
    m = re.search(r"(NCT\d+)", s or "")
    return m.group(1) if m else None


class Resolver:
    """Drug name -> intervention_ids, over the chembl-tagged drug name space."""
    def __init__(self, con):
        rows = con.execute("SELECT intervention_id, intervention_name, chembl_id FROM interventions "
                           "WHERE intervention_name IS NOT NULL AND chembl_id IS NOT NULL").fetchall()
        self.ids = [r[0] for r in rows]
        self.names = [r[1] for r in rows]
        self.by_name = defaultdict(list)
        self.by_chembl = defaultdict(list)
        self.id2chembl = {}
        for iid, nm, ch in rows:
            self.by_name[nm.lower()].append(iid)
            self.by_chembl[ch].append(iid)
            self.id2chembl[iid] = ch

    def exact(self, name):
        return list(self.by_name.get((name or "").lower(), []))

    def fuzzy(self, name, k=5, cutoff=85):
        if not HAVE_RF:
            return self.exact(name)
        ms = process.extract(name, self.names, scorer=fuzz.ratio, limit=k, score_cutoff=cutoff)
        return [self.ids[m[2]] for m in ms]

    def synonym(self, name, k=5, cutoff=85):
        seed = self.fuzzy(name, k=k, cutoff=cutoff) or self.exact(name)
        ids = set(seed)
        for i in seed:
            ch = self.id2chembl.get(i)
            if ch:
                ids.update(self.by_chembl.get(ch, []))
        return list(ids)

    def resolve(self, tier, name):
        return {"exact": self.exact, "fuzzy": self.fuzzy, "synonym": self.synonym}[tier](name)


def failures_for(con, ids, condition_id, inject_nct=None):
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    rows = con.execute(
        "SELECT source_record_id, primary_endpoint_met, p_value_primary, publication_year, failure_category "
        f"FROM trial_failure_results WHERE condition_id=? AND intervention_id IN ({ph}) AND source_record_id LIKE '%NCT%'",
        [condition_id] + list(ids)).fetchall()
    out = []
    for r in rows:
        nct = extract_nct(r[0])
        pv = 14.73 if (inject_nct and nct == inject_nct) else r[2]
        out.append({"nct": nct, "endpoint_met": r[1], "p_value": pv, "year": r[3], "failure_category": r[4]})
    return out


def others_count(con, ids, condition_id):
    if not ids:
        return 0
    ph = ",".join("?" * len(ids))
    return con.execute(f"SELECT COUNT(*) FROM trial_failure_results WHERE condition_id!=? AND intervention_id IN ({ph})",
                       [condition_id] + list(ids)).fetchone()[0]


def det_policy_on(recs, has_other, inject):
    """Best-possible routing GIVEN what the tier retrieved -> the per-tier ceiling."""
    eff = [r for r in recs if r["failure_category"] == "efficacy"]
    if eff and inject:
        return {"action": "flag", "cited_nct": None}
    met = {r["endpoint_met"] for r in recs if r["endpoint_met"] is not None}
    if len(met) > 1:
        return {"action": "reject", "cited_nct": None}
    if eff:
        return {"action": "ground", "cited_nct": eff[0]["nct"]}
    return {"action": "verify" if has_other else "defer", "cited_nct": None}


def score(dec, sk):
    a, ga = dec["action"], sk["gold_action"]
    if ga == "ground":
        return a == "ground" and dec.get("cited_nct") == sk["gold_nct"]
    if a == "ground":
        return False
    return a == ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_fuzzy.jsonl"))
    a = ap.parse_args()
    con = sqlite3.connect(CT_DB)
    R = Resolver(con)
    tasks = [json.loads(l) for l in open(a.tasks)]
    print(f"rapidfuzz: {HAVE_RF} | tasks: {len(tasks)} | resolver name-space: {len(R.ids)} drug forms\n")

    for tier in ["exact", "fuzzy", "synonym"]:
        bycls = defaultdict(lambda: [0, 0])
        ground_recall = [0, 0]   # cross-form ground: did the recorded failure surface?
        for t in tasks:
            o, sk = t["observation"], t["scoring_key"]
            ids = R.resolve(tier, o["query_name"])
            recs = failures_for(con, ids, o["condition_id"], sk["gold_nct"] if sk["inject_impossible_value"] else None)
            dec = det_policy_on(recs, others_count(con, ids, o["condition_id"]) > 0, sk["inject_impossible_value"])
            ok = score(dec, sk)
            bycls[t["action_class"]][0] += ok
            bycls[t["action_class"]][1] += 1
            if t["action_class"] == "ground" and o["is_cross_form"]:
                ground_recall[1] += 1
                ground_recall[0] += any(r["failure_category"] == "efficacy" for r in recs)
        tot = sum(v[1] for v in bycls.values()); cor = sum(v[0] for v in bycls.values())
        gr = ground_recall[0] / ground_recall[1] if ground_recall[1] else float("nan")
        print(f"=== tier={tier:7s} | det-policy ceiling acc {cor}/{tot}={cor/tot:.3f} | "
              f"cross-form ground-recall {ground_recall[0]}/{ground_recall[1]}={gr:.3f} ===")
        for c, (k, n) in sorted(bycls.items()):
            print(f"     {c:7s}: {k}/{n}={k/n:.2f}")


if __name__ == "__main__":
    main()
