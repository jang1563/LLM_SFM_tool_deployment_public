"""DrugResolver — free-text drug name -> chembl_id (+ abstain), then same-chembl failure grounding.
A2 drug leg with CURRENT resources (no external KB yet): dosage/salt/formulation normalize -> exact
in-registry chembl match -> in-registry SapBERT semantic match -> abstain. Extensible: add a ChEMBL
molecule_synonyms tier + an RxNorm approx-match tier (research codes / brand / lay) when those land."""
import re, sqlite3, os, json, numpy as np, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from paths import A2_DIR, CT_DB
A2 = str(A2_DIR)
CT = str(CT_DB)
DOSAGE = re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|ug|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w*|part\s*\d+|cohort\s*\w+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|injection|capsules?|tablet|for inhalation|q\dw|once weekly|bid|qd)\b", re.I)
SALT = re.compile(r"\b(sodium|hydrochloride|hcl|sulfate|sulphate|mesylate|bitartrate|tartrate|acetate|citrate|phosphate|maleate|fumarate|succinate|hydrobromide|besylate|tosylate)\b", re.I)
# minerals where the 'cation' IS the active drug -> stripping the salt is WRONG (magnesium sulfate != magnesium)
CATION = {"magnesium","calcium","sodium","potassium","zinc","iron","ferrous","ferric","lithium","aluminum","aluminium","ammonium","chloride","bicarbonate","carbonate","selenium","copper","manganese","strontium"}
VAGUE = re.compile(r"placebo|comparator|standard of care|best supportive|investigational|study drug|matching|vehicle|sham|\+|/| and | plus | with |combination|regimen|background therapy", re.I)
def _clean(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()
def _norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower())
    base = _clean(DOSAGE.sub(" ", s))            # dosage/formulation only
    salted = _clean(SALT.sub(" ", base))         # also strip counter-ion salts
    return base if (not salted or salted in CATION) else salted  # keep salt if stripping leaves a bare cation

class DrugResolver:
    def __init__(self, tau=0.92):
        self.tau = tau; self.con = sqlite3.connect(CT)
        # name -> chembl_id (in-registry), and chembl -> failure conditions
        self.name2ch = {}; self.fail = set(r[0] for r in self.con.execute("SELECT DISTINCT intervention_id FROM trial_failure_results"))
        self.ch2conds = defaultdict(set)
        for iid, nm, ch in self.con.execute("SELECT intervention_id, intervention_name, chembl_id FROM interventions WHERE chembl_id IS NOT NULL AND intervention_name IS NOT NULL"):
            n = _norm(nm)
            if n: self.name2ch.setdefault(n, ch)
        for iid, cid in self.con.execute("SELECT intervention_id, condition_id FROM trial_failure_results"):
            ch = self.con.execute("SELECT chembl_id FROM interventions WHERE intervention_id=?", (iid,)).fetchone()
        # build chembl -> failure condition names (for grounding)
        for iid, ch in self.con.execute("SELECT intervention_id, chembl_id FROM interventions WHERE chembl_id IS NOT NULL"):
            if iid in self.fail:
                for (cn,) in self.con.execute("SELECT DISTINCT c.condition_name FROM trial_failure_results t JOIN conditions c ON t.condition_id=c.condition_id WHERE t.intervention_id=?", (iid,)):
                    if cn: self.ch2conds[ch].add(cn)
        self.inames = list(self.name2ch); print(f"in-registry chembl-named drugs: {len(self.inames)}")
        # curated ChEMBL alias KB (brand / research-code / INN ...) -> chembl_id  (the measured drug lever)
        self.alias = json.load(open(A2 + "/chembl_aliases.json")) if os.path.exists(A2 + "/chembl_aliases.json") else {}
        print(f"ChEMBL alias KB: {len(self.alias)} names")
        # SapBERT index (cache)
        dev = "mps"; M = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.tok = AutoTokenizer.from_pretrained(M); self.model = AutoModel.from_pretrained(M).to(dev).eval(); self.dev = dev
        cache = A2 + "/drug_index.npy"
        if os.path.exists(cache): self.I = np.load(cache)
        else: print("embedding drug index...", flush=True); self.I = self._embed(self.inames); np.save(cache, self.I)

    def _embed(self, strs, bs=256):
        out = []
        for i in range(0, len(strs), bs):
            t = self.tok(strs[i:i+bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(self.dev)
            with torch.no_grad(): e = self.model(**t).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e, dim=1).cpu())
            if i % 12800 == 0 and len(strs) > 5000: print(f"  {i}/{len(strs)}", flush=True)
        return torch.cat(out).numpy().astype("float32")

    def resolve(self, text):
        if VAGUE.search(text or ""): return {"abstain": "vague/combo/placebo (non-resolvable)"}
        n = _norm(text)
        if not n: return {"abstain": "empty after normalize"}
        if n in self.name2ch: return {"chembl_id": self.name2ch[n], "matched": n, "conf": 1.0, "source": "exact"}
        if n in self.alias: return {"chembl_id": self.alias[n], "matched": n, "conf": 1.0, "source": "chembl_alias"}  # curated KB (brand/research-code)
        q = self._embed([n]); sims = (q @ self.I.T)[0]; j = int(sims.argmax()); s = float(sims[j])
        if s >= self.tau: return {"chembl_id": self.name2ch[self.inames[j]], "matched": self.inames[j], "conf": round(s, 3), "source": "sapbert"}
        return {"abstain": f"no chembl match >= {self.tau} (best {s:.2f}: {self.inames[j]}); needs RxNorm/UNII"}

    def ground(self, chembl_id):
        conds = sorted(self.ch2conds.get(chembl_id, []))
        return {"chembl_id": chembl_id, "n_failure_conditions": len(conds), "examples": conds[:5]}


if __name__ == "__main__":
    R = DrugResolver(); print("loaded.\n")
    for q in ["Degarelix 30 mg", "(Cystagon) Cysteamine Bitartrate", "Sitagliptin 100 mg",
              "Treprostinil sodium for inhalation", "Tylenol", "BI 10773", "Placebo for Voclosporin", "aspirin"]:
        r = R.resolve(q); print(f"resolve({q!r}) -> {r}")
    print("\n--- grounding demo ---")
    d = R.resolve("Sitagliptin 100 mg")
    if d.get("chembl_id"): print("ground:", R.ground(d["chembl_id"]))
