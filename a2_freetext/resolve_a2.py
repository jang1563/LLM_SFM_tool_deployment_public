"""A2Resolver — the deployable contract: free-text (drug, indication) -> recorded prior EFFICACY
failures, or an honest abstain. Composes DrugResolver (->chembl_id) + DiseaseResolver (->MONDO) and
joins the failure records by chembl AND the directional disease-hierarchy (EXACT/SUBTYPE/ANCESTOR).
This is the front-door of the grounding shim; ChEMBL/RxNorm tiers extend the drug leg's external reach."""
import sqlite3, json, re
from resolve_disease import DiseaseResolver
from resolve_drug import DrugResolver
from paths import A2_DIR, CT_DB
CT = str(CT_DB)
A2 = str(A2_DIR)

def _link_tier(match, conf):
    # precision-calibrated (eval_disease_precision.py): dict/exact + conf>=0.95 ~92% precise; 0.90-0.95 ~66%;
    # 0.85-0.90 ~50% (coin-flip). The crosswalk's sapbert_med (0.85-0.90) links must NOT pass as confident.
    if match == "dict" or (conf or 0) >= 0.95: return "confident"
    if (conf or 0) >= 0.90: return "advisory"
    return "low"

class A2Resolver:
    def __init__(self, rerank=False):
        # rerank=True routes the indication leg's 0.85-0.95 band through the validated BandReranker
        # (band precision 0.76->0.97); off by default to avoid an LLM call per ambiguous query.
        base = DiseaseResolver()
        if rerank:
            from rerank import RerankedDiseaseResolver
            self.dis = RerankedDiseaseResolver(base=base)
        else:
            self.dis = base
        self.drug = DrugResolver(); self.con = sqlite3.connect(CT)
        cx = json.load(open(A2 + "/conditions_mondo_crosswalk.json"))
        # carry the target-side link tier so a 'grounded' verdict can't silently rest on a 50%-precise link
        self.cond2mondo = {int(c): (v["mondo_id"], _link_tier(v["match"], v.get("conf"))) for c, v in cx.items()}

    def _ground_set(self, mondo_id):
        h = self.dis.hier.get(mondo_id, {})
        s = {mondo_id: "EXACT"}
        for d in h.get("desc2", []): s.setdefault(d, "SUBTYPE")
        for p in h.get("parents", []): s.setdefault(p, "ANCESTOR")
        return s

    def answer(self, drug, indication):
        dr = self.drug.resolve(drug); di = self.dis.resolve(indication)
        if "abstain" in dr: return {"verdict": "abstain", "leg": "drug", "reason": dr["abstain"]}
        if "abstain" in di: return {"verdict": "abstain", "leg": "indication", "reason": di["abstain"]}
        ch, qm = dr["chembl_id"], di["mondo_id"]; gs = self._ground_set(qm)
        # all recorded efficacy failures for this drug (any surface form sharing the chembl_id)
        rows = self.con.execute(
            "SELECT t.source_record_id, t.condition_id, c.condition_name FROM trial_failure_results t "
            "JOIN conditions c ON t.condition_id=c.condition_id JOIN interventions i ON t.intervention_id=i.intervention_id "
            "WHERE i.chembl_id=? AND t.failure_category='efficacy'", (ch,)).fetchall()
        hits = []
        for nct, cid, cname in rows:
            mt = self.cond2mondo.get(cid)
            if mt and mt[0] in gs:
                mid, link_tier = mt
                m = re.search(r"NCT\d+", nct or "")
                hits.append({"nct": m.group(0) if m else None, "condition": cname, "relation": gs[mid], "target_link": link_tier})
        # a hit is trustworthy only if BOTH the query resolution AND the target-side crosswalk link are confident
        q_conf = di.get("source") in ("dict", "sapbert_confident") or di.get("conf", 1.0) >= 0.95
        confident = [h for h in hits if h["target_link"] == "confident"]
        if not hits:
            verdict = "no_recorded_failure"
        elif confident and q_conf:
            verdict = "grounded"
        else:
            verdict = "grounded_advisory"   # rests on a 0.85-0.95 query and/or target link — verify before trusting
        return {"verdict": verdict,
                "drug": {"query": drug, "chembl_id": ch, "matched": dr.get("matched")},
                "indication": {"query": indication, "mondo": di["name"], "mondo_id": qm, "source": di.get("source")},
                "n_failures": len(hits), "n_confident": len(confident), "failures": hits[:8]}


if __name__ == "__main__":
    R = A2Resolver(); print("loaded.\n")
    # find a real (chembl, efficacy-failure, crosswalked condition) to demo grounding
    seed = R.con.execute(
        "SELECT i.chembl_id, c.condition_id, c.condition_name FROM trial_failure_results t "
        "JOIN interventions i ON t.intervention_id=i.intervention_id JOIN conditions c ON t.condition_id=c.condition_id "
        "WHERE t.failure_category='efficacy' AND i.chembl_id IS NOT NULL LIMIT 2000").fetchall()
    cm = None
    for ch, cid, cn in seed:
        if cid in R.cond2mondo: cm = (ch, cn); break
    print(f"demo seed: chembl {cm[0]} failed for {cm[1]!r}\n")
    name = R.con.execute("SELECT intervention_name FROM interventions WHERE chembl_id=? LIMIT 1", (cm[0],)).fetchone()[0]
    for dq, iq in [(name, cm[1]), (name + " 50 mg", cm[1]),
                   ("Tylenol", "heart attack"), ("aspirin", "high blood pressure"),
                   ("Placebo", cm[1]), (name, "Racial Bias")]:
        a = R.answer(dq, iq)
        v = a["verdict"]
        extra = f" ({a['n_failures']} failures, eg {a['failures'][:1]})" if v == "grounded" else (f" [{a.get('leg')}: {a.get('reason','')[:40]}]" if v == "abstain" else "")
        print(f"answer({dq!r}, {iq!r}) -> {v}{extra}")
