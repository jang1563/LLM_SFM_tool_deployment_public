"""Eval — disease-leg leave-one-out generalization recall, gold = MONDO id (NON-circular: MONDO's own
synonym grouping is the gold, independent of the resolver's crosswalk). For each MONDO concept with >=3
surface forms (name+synonyms = real lay/abbrev/exact variation), hold out a middle form, index the rest,
and ask each method to map the held form -> right MONDO id. Ladder: exact -> fuzzy -> SapBERT(semantic).
Mirrors the drug LOO; expect SapBERT to add MORE here (medical synonymy is SapBERT's training objective)."""
import re, numpy as np, torch
from rapidfuzz import fuzz, process
from transformers import AutoTokenizer, AutoModel
from paths import A2_DIR
A2 = str(A2_DIR)
GENERIC = {"disease","diseases","disorder","disorders","syndrome","syndromes","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication"}
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

# MONDO concept -> surface forms (name + EXACT/NARROW synonyms), skip obsolete + generic + too-short
id2forms = {}; cur = None; obs = False
for line in open(A2 + "/mondo.obo"):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None; obs = False
    elif line.startswith("id: "): cur = line[4:]
    elif cur and cur.startswith("MONDO:"):
        if line.startswith("is_obsolete: true"): id2forms.pop(cur, None); cur = None
        elif line.startswith("name: "):
            n = norm(line[6:])
            if len(n) >= 4 and n not in GENERIC: id2forms.setdefault(cur, set()).add(n)
        elif line.startswith("synonym: "):
            m = re.match(r'synonym: "(.*?)" (EXACT|NARROW)', line)
            if m:
                n = norm(m.group(1))
                if len(n) >= 4 and n not in GENERIC: id2forms.setdefault(cur, set()).add(n)

test = []; train = {}; held_set = set()   # train: form -> mondo_id
for mid, forms in id2forms.items():
    fs = sorted(forms)
    if len(fs) >= 3:
        held = fs[len(fs) // 2]; held_set.add((held, mid)); test.append((held, mid))
for mid, forms in id2forms.items():
    for f in forms:
        if (f, mid) in held_set: continue
        train.setdefault(f, mid)
tnorms = list(train)
print(f"held-out disease forms: {len(test)} | index forms: {len(tnorms)}", flush=True)

dev = "mps"; M = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tok = AutoTokenizer.from_pretrained(M); model = AutoModel.from_pretrained(M).to(dev).eval()
def embed(strs, bs=512):
    out = []
    for i in range(0, len(strs), bs):
        t = tok(strs[i:i+bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        with torch.no_grad(): e = model(**t).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
        if i % 25600 == 0: print(f"  embed {i}/{len(strs)}", flush=True)
    return torch.cat(out).numpy().astype("float32")
import os
ic, tc = A2 + "/disease_index.npy", A2 + "/disease_test.npy"
if os.path.exists(ic) and os.path.exists(tc) and np.load(ic).shape[0] == len(tnorms) and np.load(tc).shape[0] == len(test):
    print("loading cached embeddings...", flush=True); I = np.load(ic); Tq = np.load(tc)
else:
    print("embedding index...", flush=True); I = embed(tnorms); np.save(ic, I)
    print("embedding test...", flush=True); Tq = embed([h for h, _ in test]); np.save(tc, Tq)

FZ_SAMPLE = 1500   # rapidfuzz over 77k choices is the bottleneck; sample for an estimate
exact = fuzzy = sap = 0; resolver = 0; fuzzy_n = 0
for k, (held, mid) in enumerate(test):
    if train.get(held) == mid: exact += 1
    if k < FZ_SAMPLE:
        fuzzy_n += 1
        m = process.extractOne(held, tnorms, scorer=fuzz.token_sort_ratio)
        if m and m[1] >= 90 and train[m[0]] == mid: fuzzy += 1
    sims = (Tq[k] @ I.T); j = int(sims.argmax()); sp = float(sims[j]) >= 0.90 and train[tnorms[j]] == mid
    if sp: sap += 1
    if train.get(held) == mid or sp: resolver += 1
N = len(test)
print(f"\n=== DISEASE-LEG LOO RECALL (gold=MONDO id, held-out UNSEEN real synonyms, N={N}) ===")
print(f"  exact-string          : {exact/N:.3f}  (naive — false-deprioritizes {1-exact/N:.0%})")
print(f"  fuzzy(token>=90)      : {fuzzy/fuzzy_n:.3f}  (sampled n={fuzzy_n})")
print(f"  SapBERT(>=0.90)       : {sap/N:.3f}")
print(f"  RESOLVER(exact+SapBERT): {resolver/N:.3f}  (vs naive {exact/N:.3f} -> recovers {(resolver-exact)/N:.0%} of otherwise-missed disease matches)")
