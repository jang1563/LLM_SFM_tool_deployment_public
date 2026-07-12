"""DiseaseResolver — free-text indication -> MONDO concept (+ abstain), then hierarchy grounding.
A2 disease leg: dictionary-NER + generic-blocklist + obsolete-filter + SapBERT residue + antonym-veto
+ two-threshold abstain; grounding via Q ∪ descendants(≤2) ∪ immediate-parent with relation tags."""
import re, json, numpy as np, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from paths import A2_DIR
A2 = str(A2_DIR)
GENERIC = {"disease","diseases","disorder","disorders","syndrome","syndromes","syndromic disease","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication","infectious disease","rare disease"}
def _norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# Antonym veto = fire ONLY when query and candidate sit on OPPOSITE sides of an antonym pair (both present,
# different member). The old code vetoed on a SYMMETRIC-DIFFERENCE of substring tokens, so a missing token
# false-vetoed correct synonyms (hyperthyroidism->thyrotoxicosis: qneg={hyper} XOR {} -> wrong veto).
ANTONYM_PAIRS = [(r"\bhyper", r"\bhypo"), (r"\btype 1\b", r"\btype 2\b"),
                 (r"\bbenign\b", r"\bmalignant\b"), (r"\bacute\b", r"\bchronic\b"),
                 (r"\bprimary\b", r"\bsecondary\b")]
def _antonym_conflict(q, c):
    q, c = _norm(q), _norm(c)
    for a, b in ANTONYM_PAIRS:
        if (re.search(a, q) and re.search(b, c)) or (re.search(b, q) and re.search(a, c)): return True
    qnh, cnh = "non hodgkin" in q, "non hodgkin" in c          # non-hodgkin contains 'hodgkin' -> handle explicitly
    qh, ch = ("hodgkin" in q) and not qnh, ("hodgkin" in c) and not cnh
    return (qnh and ch) or (qh and cnh)

class DiseaseResolver:
    # precision-calibrated (eval_disease_precision.py): >=0.95 is 92% precise (CONFIDENT); 0.90-0.95 is
    # only 66% (34% false-ground -> ADVISORY, needs reranker/verify); <0.90 abstain (0.85-0.90 was a 50% coin-flip).
    def __init__(self, tau_high=0.95, tau_med=0.90):
        self.tau_high, self.tau_med = tau_high, tau_med
        H = json.load(open(A2 + "/mondo_hierarchy.json")); self.hier, self.idname = H["hier"], H["names"]
        cx = json.load(open(A2 + "/conditions_mondo_crosswalk.json"))
        self.mondo2conds = defaultdict(list)
        for cid, v in cx.items(): self.mondo2conds[v["mondo_id"]].append(v["condition_name"])
        # MONDO label->id (blocklist + skip obsolete), aligned with the cached SapBERT index
        self.lab2id = {}; cur = None
        for line in open(A2 + "/mondo.obo"):
            line = line.rstrip("\n")
            if line == "[Term]": cur = None
            elif line.startswith("id: "): cur = line[4:]
            elif cur and cur.startswith("MONDO:"):
                if line.startswith("name: "):
                    nm = _norm(line[6:])
                    if len(nm) >= 4 and nm not in GENERIC and cur != "MONDO:0000001": self.lab2id.setdefault(nm, cur)
                elif line.startswith("synonym: "):
                    m = re.match(r'synonym: "(.*?)" (EXACT|NARROW)', line)
                    if m:
                        t = _norm(m.group(1))
                        if len(t) >= 4 and t not in GENERIC and cur != "MONDO:0000001": self.lab2id.setdefault(t, cur)
        self.labels = sorted(self.lab2id); self.labset = set(self.lab2id)
        self.Lmat = np.load(A2 + "/mondo_sapbert.npy"); assert len(self.labels) == self.Lmat.shape[0]
        dev = "mps"; M = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.tok = AutoTokenizer.from_pretrained(M); self.model = AutoModel.from_pretrained(M).to(dev).eval(); self.dev = dev

    def _obsolete(self, mid): return self.hier.get(mid, {}).get("obsolete", False)

    def _embed(self, s):
        t = self.tok([s], padding=True, truncation=True, max_length=32, return_tensors="pt").to(self.dev)
        with torch.no_grad(): e = self.model(**t).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype("float32")

    def resolve(self, text):
        """free text -> {mondo_id,name,conf,source} or {abstain:reason}"""
        toks = _norm(text).split()
        # 1) dictionary longest-substring (skip obsolete)
        best = None
        for n in range(min(6, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i+n])
                if g in self.labset and not self._obsolete(self.lab2id[g]) and (best is None or len(g) > len(best)): best = g
            if best: break
        if best: return {"mondo_id": self.lab2id[best], "name": self.idname.get(self.lab2id[best]), "conf": 1.0, "source": "dict"}
        # 2) SapBERT top-k with antonym-aware pick + obsolete skip
        q = self._embed(_norm(text)); sims = (q @ self.Lmat.T)[0]; order = np.argsort(-sims)[:10]
        for j in order:
            mid = self.lab2id[self.labels[j]]; sim = float(sims[j])
            if self._obsolete(mid): continue
            if _antonym_conflict(text, self.labels[j]):  # opposite side of an antonym pair -> veto, try next
                continue
            if sim >= self.tau_high: return {"mondo_id": mid, "name": self.idname.get(mid), "conf": round(sim,3), "source": "sapbert_confident"}  # >=0.95, 92% precise
            if sim >= self.tau_med:  return {"mondo_id": mid, "name": self.idname.get(mid), "conf": round(sim,3), "source": "sapbert_advisory", "advisory": True, "needs_rerank": True}  # 0.90-0.95, 66% precise
            break
        return {"abstain": f"no MONDO match above {self.tau_med} (best {float(sims[order[0]]):.2f})"}

    def ground(self, mondo_id):
        """recorded failure conditions that match the query disease, with relation tags."""
        h = self.hier.get(mondo_id, {}); out = []
        for mid, rel in [(mondo_id, "EXACT")] + [(d, "SUBTYPE") for d in h.get("desc2", [])] + [(p, "ANCESTOR") for p in h.get("parents", [])]:
            conds = self.mondo2conds.get(mid)
            if conds: out.append({"mondo_id": mid, "name": self.idname.get(mid), "relation": rel, "n_failure_conditions": len(conds), "examples": conds[:3]})
        return out


if __name__ == "__main__":
    R = DiseaseResolver(); print("loaded.\n")
    for q in ["heart attack", "STEMI", "high blood pressure", "type 2 diabetes mellitus",
              "non-small cell lung cancer", "Tylenol", "Racial Bias", "Alzheimers"]:
        r = R.resolve(q)
        print(f"resolve({q!r}) -> {r}")
    print("\n--- grounding demo ---")
    mi = R.resolve("heart attack")
    if mi.get("mondo_id"):
        g = R.ground(mi["mondo_id"])
        print(f"ground('heart attack' = {mi['mondo_id']} {mi['name']}): {len(g)} matching disease nodes")
        for x in g[:6]: print(f"  [{x['relation']:8s}] {x['name'][:34]:34s} n={x['n_failure_conditions']} eg={x['examples'][:1]}")
