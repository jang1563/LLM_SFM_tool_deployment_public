import re, sqlite3, random, os, numpy as np, torch
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
labels=sorted(lab2id); print(f"MONDO labels to index: {len(labels)}", flush=True)
con=sqlite3.connect(str(CT_DB))
fail=set(r[0] for r in con.execute("SELECT DISTINCT condition_id FROM trial_failure_results"))
rows=[(c,n) for c,n in con.execute("SELECT condition_id, condition_name FROM conditions WHERE condition_name IS NOT NULL") if c in fail]
labset=set(lab2id)
def dict_hit(nm):
    toks=norm(nm).split()
    for n in range(min(6,len(toks)),0,-1):
        for i in range(len(toks)-n+1):
            if " ".join(toks[i:i+n]) in labset: return True
    return False
residue=[(c,n) for c,n in rows if not dict_hit(n)]
print(f"failure-bearing residue (dict-abstain): {len(residue)}", flush=True)
random.seed(7); random.shuffle(residue); samp=residue[:250]
dev="mps"; M="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
print("loading SapBERT...", flush=True)
tok=AutoTokenizer.from_pretrained(M); model=AutoModel.from_pretrained(M).to(dev).eval()
def embed(strs,bs=256):
    out=[]
    for i in range(0,len(strs),bs):
        t=tok(strs[i:i+bs],padding=True,truncation=True,max_length=32,return_tensors="pt").to(dev)
        with torch.no_grad(): e=model(**t).last_hidden_state[:,0]
        out.append(torch.nn.functional.normalize(e,dim=1).cpu())
        if i % 12800==0: print(f"  embedded {i}/{len(strs)}",flush=True)
    return torch.cat(out).numpy().astype("float32")
cache=A2+"/mondo_sapbert.npy"
if os.path.exists(cache): Lmat=np.load(cache); print("loaded cached MONDO emb",Lmat.shape,flush=True)
else:
    print("embedding MONDO index...",flush=True); Lmat=embed(labels); np.save(cache,Lmat); print("cached.",flush=True)
Qmat=embed([norm(n) for _,n in samp])
sims=Qmat@Lmat.T; top=sims.argmax(1); topsim=sims.max(1)
b=np.array(topsim)
print(f"\nSIM DIST (of {len(samp)} residue): >=0.95 {int((b>=0.95).sum())} | 0.90-0.95 {int(((b>=0.90)&(b<0.95)).sum())} | 0.85-0.90 {int(((b>=0.85)&(b<0.90)).sum())} | 0.80-0.85 {int(((b>=0.80)&(b<0.85)).sum())} | <0.80 {int((b<0.80).sum())}", flush=True)
print("(>=~0.90 high-confidence recovery; <0.80 likely abstain/non-disease)\n", flush=True)
print("=== nearest (cond -> MONDO | sim), sorted by sim ===", flush=True)
order=np.argsort(-b)
idxs=list(order[:11])+list(order[len(order)//2-3:len(order)//2+3])+list(order[-8:])
for j in idxs:
    c,n=samp[j]; print(f"  [{topsim[j]:.2f}] {n[:44]:44s} -> {id2name.get(lab2id[labels[top[j]]],'?')[:30]}", flush=True)
