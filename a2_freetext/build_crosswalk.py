import re, sqlite3, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from paths import A2_DIR, CT_DB
A2=str(A2_DIR)
def norm(s):
    s=re.sub(r"\(.*?\)"," ",(s or "").lower()); s=re.sub(r"[^a-z0-9 ]+"," ",s); return re.sub(r"\s+"," ",s).strip()
GENERIC={"disease","diseases","disorder","disorders","syndrome","syndromes","syndromic disease","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication","infectious disease","rare disease"}
lab2id={}; id2name={}; cid=None
for line in open(A2+"/mondo.obo"):
    line=line.rstrip("\n")
    if line=="[Term]": cid=None
    elif line.startswith("id: "): cid=line[4:]
    elif cid and cid.startswith("MONDO:"):
        if line.startswith("name: "):
            raw=line[6:]; nm=norm(raw); id2name.setdefault(cid,raw)
            if len(nm)>=4 and nm not in GENERIC and cid!="MONDO:0000001": lab2id.setdefault(nm,cid)
        elif line.startswith("synonym: "):
            m=re.match(r'synonym: "(.*?)" (EXACT|NARROW)',line)
            if m:
                t=norm(m.group(1))
                if len(t)>=4 and t not in GENERIC and cid!="MONDO:0000001": lab2id.setdefault(t,cid)
labels=sorted(lab2id); labset=set(lab2id)
Lmat=np.load(A2+"/mondo_sapbert.npy")
assert len(labels)==Lmat.shape[0], f"cache mismatch {len(labels)} vs {Lmat.shape}"
print(f"MONDO labels {len(labels)} aligned with cache {Lmat.shape}", flush=True)
con=sqlite3.connect(str(CT_DB))
fail=set(r[0] for r in con.execute("SELECT DISTINCT condition_id FROM trial_failure_results"))
rows=con.execute("SELECT condition_id, condition_name FROM conditions WHERE condition_name IS NOT NULL").fetchall()
def dict_resolve(nm):
    toks=norm(nm).split(); best=None
    for n in range(min(6,len(toks)),0,-1):
        for i in range(len(toks)-n+1):
            g=" ".join(toks[i:i+n])
            if g in labset and (best is None or len(g)>len(best)): best=g
        if best: break
    return lab2id[best] if best else None
cross={}; residue=[]
for c,n in rows:
    mid=dict_resolve(n)
    if mid: cross[c]={"mondo_id":mid,"mondo_name":id2name.get(mid),"match":"dict","conf":1.0,"condition_name":n}
    else: residue.append((c,n))
print(f"dict-resolved {len(cross)}; residue {len(residue)} -> SapBERT", flush=True)
# SapBERT the residue
dev="mps"; M="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tok=AutoTokenizer.from_pretrained(M); model=AutoModel.from_pretrained(M).to(dev).eval()
def embed(strs,bs=256):
    out=[]
    for i in range(0,len(strs),bs):
        t=tok(strs[i:i+bs],padding=True,truncation=True,max_length=32,return_tensors="pt").to(dev)
        with torch.no_grad(): e=model(**t).last_hidden_state[:,0]
        out.append(torch.nn.functional.normalize(e,dim=1).cpu())
        if i%12800==0: print(f"  residue embed {i}/{len(strs)}",flush=True)
    return torch.cat(out).numpy().astype("float32")
Q=embed([norm(n) for _,n in residue])
# chunked ANN to bound memory
for s in range(0,len(residue),4096):
    sims=Q[s:s+4096]@Lmat.T; ti=sims.argmax(1); tv=sims.max(1)
    for k,(c,n) in enumerate(residue[s:s+4096]):
        sim=float(tv[k]); mid=lab2id[labels[ti[k]]]
        if sim>=0.90: cross[c]={"mondo_id":mid,"mondo_name":id2name.get(mid),"match":"sapbert_high","conf":round(sim,3),"condition_name":n}
        elif sim>=0.85: cross[c]={"mondo_id":mid,"mondo_name":id2name.get(mid),"match":"sapbert_med","conf":round(sim,3),"condition_name":n}
        # else abstain (not added)
json.dump(cross, open(A2+"/conditions_mondo_crosswalk.json","w"))
from collections import Counter
mt=Counter(v["match"] for v in cross.values())
N=len(rows); fcov=sum(1 for c in fail if c in cross); fn=len(fail)
print(f"\n=== CROSSWALK BUILT -> conditions_mondo_crosswalk.json ===")
print(f"  ALL conditions: {len(cross)}/{N} = {100*len(cross)/N:.1f}% resolved (by match: {dict(mt)})")
print(f"  FAILURE-bearing: {fcov}/{fn} = {100*fcov/fn:.1f}% resolved | abstain {100*(fn-fcov)/fn:.1f}%")
