"""Drug LOO, KB arm — does the curated ChEMBL alias index close the 39% gap SapBERT couldn't?
Same held-out test set as eval_drug_loo.py (real registry surface forms, gold=chembl). No embedding
(pure dict lookup) so it doesn't contend with the disease-LOO MPS job. Ladder vs the prior SapBERT run."""
import re, sqlite3, json
from collections import defaultdict
from paths import A2_DIR, CT_DB
CT = str(CT_DB)
A2 = str(A2_DIR)
DOSE = re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|ug|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w*|part\s*\d+|cohort\s*\w+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|sodium|hydrochloride|hcl|sulfate|mesylate|bitartrate|acetate|injection|capsules?|tablet|for inhalation|q\dw|once weekly|bid|qd)\b", re.I)
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = DOSE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

alias = json.load(open(A2 + "/chembl_aliases.json"))   # curated name -> chembl
con = sqlite3.connect(CT)
fail = set(r[0] for r in con.execute("SELECT DISTINCT intervention_id FROM trial_failure_results"))
ch2raw = defaultdict(set)
for iid, nm, ch in con.execute("SELECT intervention_id, intervention_name, chembl_id FROM interventions WHERE chembl_id IS NOT NULL AND intervention_name IS NOT NULL"):
    if iid in fail and nm.strip(): ch2raw[ch].add(nm.strip())
# rebuild the SAME held-out test set: middle form for chembls with >=3 distinct raw forms
test = []; train_raw = {}; train_lower = {}; held_set = set()
for ch, raws in ch2raw.items():
    rs = sorted(raws)
    if len(rs) >= 3: held = rs[len(rs)//2]; held_set.add((held, ch)); test.append((held, ch))
for ch, raws in ch2raw.items():
    for r in raws:
        if (r, ch) in held_set: continue
        n = norm(r)
        if n: train_raw.setdefault(n, ch); train_lower.setdefault(r.lower(), ch)
N = len(test)
exact_raw = norm_hit = kb = norm_or_kb = 0
for held, ch in test:
    n = norm(held)
    er = train_lower.get(held.lower()) == ch
    nh = train_raw.get(n) == ch
    kh = alias.get(n) == ch                          # the curated KB arm
    if er: exact_raw += 1
    if nh: norm_hit += 1
    if kh: kb += 1
    if nh or kh: norm_or_kb += 1
print(f"=== DRUG-LEG LOO — KB arm (gold=chembl, held-out UNSEEN real surface forms, N={N}) ===")
print(f"  exact-raw-string        : {exact_raw/N:.3f}")
print(f"  +dosage-normalize       : {norm_hit/N:.3f}")
print(f"  ChEMBL-alias-KB (alone) : {kb/N:.3f}")
print(f"  normalize ∪ KB          : {norm_or_kb/N:.3f}   <-- vs prior normalize∪SapBERT 0.609")
print(f"  -> KB lifts normalize {norm_hit/N:.3f} -> {norm_or_kb/N:.3f}  (+{(norm_or_kb-norm_hit)/N:.0%})")
