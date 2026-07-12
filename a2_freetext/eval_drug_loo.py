"""Eval — drug-leg leave-one-out generalization recall on REAL registry surface variation (non-synthetic).
For each failure-bearing drug (chembl) with >=3 distinct raw surface forms, HOLD OUT the noisiest form,
build the resolver index on the rest, and ask each method to recover the held-out form -> right chembl.
Baseline ladder: exact-raw-string -> exact-after-dosage-normalize -> fuzzy(token) -> SapBERT(semantic).
The held-out form is UNSEEN by the index, so this measures generalization, not memorization. gold=chembl.
recall = 1 - false-deprioritization (a drug failure the method would MISS because the name didn't match)."""
import re, sqlite3, numpy as np, torch
from collections import defaultdict
from rapidfuzz import fuzz, process
from transformers import AutoTokenizer, AutoModel
from paths import CT_DB
CT = str(CT_DB)
DOSE = re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|ug|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w*|part\s*\d+|cohort\s*\w+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|sodium|hydrochloride|hcl|sulfate|mesylate|bitartrate|acetate|injection|capsules?|tablet|for inhalation|q\dw|once weekly|bid|qd)\b", re.I)
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = DOSE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

con = sqlite3.connect(CT)
fail = set(r[0] for r in con.execute("SELECT DISTINCT intervention_id FROM trial_failure_results"))
ch2raw = defaultdict(set)
for iid, nm, ch in con.execute("SELECT intervention_id, intervention_name, chembl_id FROM interventions WHERE chembl_id IS NOT NULL AND intervention_name IS NOT NULL"):
    if iid in fail and nm.strip(): ch2raw[ch].add(nm.strip())
# LOO: for chembls with >=3 distinct raw forms, hold out a deterministic MIDDLE form (real surface variation,
# not cherry-picked-hardest). The held RAW string is excluded from every index; siblings stay in.
test = []          # (held_raw, chembl)
train_raw = {}     # norm_form -> chembl   (resolver exact-norm + sapbert index)
train_lower = {}   # raw.lower() -> chembl (naive exact-raw baseline)
held_set = set()
for ch, raws in ch2raw.items():
    rs = sorted(raws)
    if len(rs) >= 3:
        held = rs[len(rs) // 2]; held_set.add((held, ch)); test.append((held, ch))
for ch, raws in ch2raw.items():
    for r in raws:
        if (r, ch) in held_set: continue          # never index a held-out string
        n = norm(r)
        if n: train_raw.setdefault(n, ch); train_lower.setdefault(r.lower(), ch)
train_norms = list(train_raw)
print(f"held-out test forms: {len(test)} | index forms: {len(train_norms)}", flush=True)

dev = "mps"; M = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tok = AutoTokenizer.from_pretrained(M); model = AutoModel.from_pretrained(M).to(dev).eval()
def embed(strs, bs=256):
    out = []
    for i in range(0, len(strs), bs):
        t = tok(strs[i:i+bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        with torch.no_grad(): e = model(**t).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy().astype("float32")
print("embedding index + test...", flush=True)
I = embed(train_norms); Tq = embed([norm(h) for h, _ in test])

exact_raw = exact_norm = fuzzy = sap = 0
for k, (held, ch) in enumerate(test):
    # 1) exact raw string (naive)
    if train_lower.get(held.lower()) == ch: exact_raw += 1
    # 2) exact after dosage-normalize
    if train_raw.get(norm(held)) == ch: exact_norm += 1
    # 3) fuzzy token match
    m = process.extractOne(norm(held), train_norms, scorer=fuzz.token_sort_ratio)
    if m and m[1] >= 90 and train_raw[m[0]] == ch: fuzzy += 1
    # 4) SapBERT
    sims = (Tq[k] @ I.T); j = int(sims.argmax())
    if float(sims[j]) >= 0.92 and train_raw[train_norms[j]] == ch: sap += 1
N = len(test)
print(f"\n=== DRUG-LEG LOO RECALL (gold=chembl, held-out UNSEEN real surface forms, N={N}) ===")
print(f"  exact-raw-string      : {exact_raw/N:.3f}  (naive baseline — false-deprioritizes {1-exact_raw/N:.0%})")
print(f"  +dosage-normalize     : {exact_norm/N:.3f}")
print(f"  +fuzzy(token>=90)     : {max(exact_norm,fuzzy)/N:.3f}  (fuzzy-only {fuzzy/N:.3f})")
print(f"  +SapBERT(>=0.92)      : {max(exact_norm,sap)/N:.3f}  (sapbert-only {sap/N:.3f})")
union = sum(1 for k,(held,ch) in enumerate(test) if train_raw.get(norm(held))==ch or (float((Tq[k]@I.T).max())>=0.92 and train_raw[train_norms[int((Tq[k]@I.T).argmax())]]==ch))
print(f"  RESOLVER(norm+SapBERT): {union/N:.3f}  (vs naive {exact_raw/N:.3f} -> recovers {(union-exact_raw)/N:.0%} of otherwise-missed drug failures)")
