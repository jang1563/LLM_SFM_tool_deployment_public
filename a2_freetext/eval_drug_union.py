"""Drug LOO — full union ladder: naive -> normalize -> +KB -> +SapBERT -> RESOLVER(all).
Same held-out test set as eval_drug_loo.py (gold=chembl). Completes the drug-leg measured ladder."""
import re, sqlite3, json, numpy as np, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from paths import A2_DIR, CT_DB
CT = str(CT_DB)
A2 = str(A2_DIR)
DOSE = re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|ug|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w*|part\s*\d+|cohort\s*\w+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|sodium|hydrochloride|hcl|sulfate|mesylate|bitartrate|acetate|injection|capsules?|tablet|for inhalation|q\dw|once weekly|bid|qd)\b", re.I)
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = DOSE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

alias = json.load(open(A2 + "/chembl_aliases.json"))
con = sqlite3.connect(CT)
fail = set(r[0] for r in con.execute("SELECT DISTINCT intervention_id FROM trial_failure_results"))
ch2raw = defaultdict(set)
for iid, nm, ch in con.execute("SELECT intervention_id, intervention_name, chembl_id FROM interventions WHERE chembl_id IS NOT NULL AND intervention_name IS NOT NULL"):
    if iid in fail and nm.strip(): ch2raw[ch].add(nm.strip())
test = []; train_raw = {}; held_set = set()
for ch, raws in ch2raw.items():
    rs = sorted(raws)
    if len(rs) >= 3: held = rs[len(rs)//2]; held_set.add((held, ch)); test.append((held, ch))
for ch, raws in ch2raw.items():
    for r in raws:
        if (r, ch) in held_set: continue
        n = norm(r)
        if n: train_raw.setdefault(n, ch)
tn = list(train_raw); N = len(test)
dev = "mps"; M = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tok = AutoTokenizer.from_pretrained(M); model = AutoModel.from_pretrained(M).to(dev).eval()
def embed(strs, bs=256):
    out = []
    for i in range(0, len(strs), bs):
        t = tok(strs[i:i+bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        with torch.no_grad(): e = model(**t).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy().astype("float32")
print(f"test {N} | index {len(tn)} — embedding...", flush=True)
I = embed(tn); Tq = embed([norm(h) for h, _ in test]); tn_ch = [train_raw[t] for t in tn]
nm_hit = kb_hit = sap_hit = uni = 0
for k, (held, ch) in enumerate(test):
    n = norm(held)
    a = train_raw.get(n) == ch                                   # normalize
    b = alias.get(n) == ch                                       # KB
    j = int((Tq[k] @ I.T).argmax()); c = float((Tq[k] @ I.T).max()) >= 0.92 and tn_ch[j] == ch  # SapBERT
    nm_hit += a; kb_hit += b; sap_hit += c
    if a or b or c: uni += 1
print(f"\n=== DRUG-LEG LOO — FULL UNION LADDER (gold=chembl, N={N}) ===")
print(f"  naive exact-raw        : 0.000")
print(f"  +dosage-normalize      : {nm_hit/N:.3f}")
print(f"  +SapBERT               : {(sum(1 for k,(h,ch) in enumerate(test) if train_raw.get(norm(h))==ch or (float((Tq[k]@I.T).max())>=0.92 and tn_ch[int((Tq[k]@I.T).argmax())]==ch)))/N:.3f}")
print(f"  +ChEMBL-KB             : {(sum(1 for k,(h,ch) in enumerate(test) if train_raw.get(norm(h))==ch or alias.get(norm(h))==ch))/N:.3f}")
print(f"  RESOLVER (norm∪KB∪SapBERT): {uni/N:.3f}   <-- deployable drug recall (RxNorm/UNII would add more)")
