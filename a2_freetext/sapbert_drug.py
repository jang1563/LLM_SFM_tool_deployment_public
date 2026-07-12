import re, sqlite3, random, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from paths import CT_DB
DOSE=re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w+|part\s*\d+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|placebo|sodium|hydrochloride|hcl|sulfate|mesylate|injection|capsules?|tablet|for inhalation)\b", re.I)
def norm(s):
    s=re.sub(r"\(.*?\)"," ",(s or "").lower()); s=DOSE.sub(" ",s); s=re.sub(r"[^a-z0-9 ]+"," ",s); return re.sub(r"\s+"," ",s).strip()
VAGUE=re.compile(r"placebo|comparator|standard of care|best supportive|investigational|study drug|matching|vehicle|sham|\+|/| and |combination|regimen|background therapy", re.I)
con=sqlite3.connect(str(CT_DB))
# index: in-registry chembl-named drugs (name -> chembl_id), all
idx=[(norm(n),ch) for n,ch in con.execute("SELECT intervention_name, chembl_id FROM interventions WHERE chembl_id IS NOT NULL AND intervention_name IS NOT NULL") if norm(n)]
# dedup by normalized name
seen={};
for nm,ch in idx: seen.setdefault(nm,ch)
inames=list(seen); print(f"chembl-named drug index: {len(inames)} unique normalized names", flush=True)
# residue: no-chembl drug/biologic failure-bearing, exclude vague
fail=set(r[0] for r in con.execute("SELECT DISTINCT intervention_id FROM trial_failure_results"))
res=[n for c,n in con.execute("SELECT intervention_id,intervention_name FROM interventions WHERE intervention_type IN ('drug','biologic') AND chembl_id IS NULL AND intervention_name IS NOT NULL") if c in fail and not VAGUE.search(n or "") and norm(n)]
random.seed(4); random.shuffle(res); samp=res[:250]
print(f"residue (no-chembl, non-vague) sample: {len(samp)} of {len(res)}", flush=True)
dev="mps"; M="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tok=AutoTokenizer.from_pretrained(M); model=AutoModel.from_pretrained(M).to(dev).eval()
def embed(strs,bs=256):
    out=[]
    for i in range(0,len(strs),bs):
        t=tok(strs[i:i+bs],padding=True,truncation=True,max_length=32,return_tensors="pt").to(dev)
        with torch.no_grad(): e=model(**t).last_hidden_state[:,0]
        out.append(torch.nn.functional.normalize(e,dim=1).cpu())
    return torch.cat(out).numpy().astype("float32")
print("embedding index...",flush=True); I=embed(inames)
Q=embed([norm(n) for n in samp])
sims=Q@I.T; top=sims.argmax(1); ts=sims.max(1); b=np.array(ts)
print(f"\nSIM DIST (of {len(samp)}): >=0.95 {int((b>=0.95).sum())} | 0.90-0.95 {int(((b>=0.90)&(b<0.95)).sum())} | 0.85-0.90 {int(((b>=0.85)&(b<0.90)).sum())} | <0.85 {int((b<0.85).sum())}", flush=True)
print("(>=0.90 = same molecule as an in-registry chembl drug, recovered WITHOUT external KB)\n", flush=True)
o=np.argsort(-b); idxs=list(o[:12])+list(o[-8:])
for j in idxs:
    print(f"  [{ts[j]:.2f}] {samp[j][:46]:46s} -> {inames[top[j]][:28]}  (chembl {seen[inames[top[j]]]})", flush=True)
