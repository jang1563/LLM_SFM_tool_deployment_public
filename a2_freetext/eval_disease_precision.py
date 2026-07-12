"""Disease PRECISION / false-grounding — the missing half of the eval + the reranker gate.
Reuses the cached SapBERT LOO embeddings (no re-embed). Two measurements, both non-circular (gold=MONDO id):
 (A) IN-BAND PRECISION: of the resolver's positives (nearest concept at sim s), what fraction are the GOLD
     concept, stratified by confidence band — quantifies the 0.85-0.90 sibling-drift; 1-precision = false-ground.
 (B) NIL-REJECTION: hold out WHOLE concepts (mask all their index forms) so the true answer is ABSENT; a query
     of a held concept SHOULD abstain. false-ground = it instead resolves to a (wrong) non-held concept at sim>=thr.
     This is the deployment-critical precision (confidently-wrong on out-of-registry input)."""
import re, numpy as np
from paths import A2_DIR
A2 = str(A2_DIR)
GENERIC = {"disease","diseases","disorder","disorders","syndrome","syndromes","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication"}
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

# rebuild the identical split (deterministic) to align with the cached embeddings
id2forms = {}; cur = None
for line in open(A2 + "/mondo.obo"):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
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
test = []; train = {}; held_set = set()
for mid, forms in id2forms.items():
    fs = sorted(forms)
    if len(fs) >= 3: held = fs[len(fs)//2]; held_set.add((held, mid)); test.append((held, mid))
for mid, forms in id2forms.items():
    for f in forms:
        if (f, mid) in held_set: continue
        train.setdefault(f, mid)
tnorms = list(train); tn_mid = np.array([train[t] for t in tnorms])
gold = np.array([mid for _, mid in test])
I = np.load(A2 + "/disease_index.npy"); Tq = np.load(A2 + "/disease_test.npy")
assert I.shape[0] == len(tnorms) and Tq.shape[0] == len(test), "cache mismatch — re-run eval_disease_loo.py"

# (A) IN-BAND PRECISION
print(f"=== (A) DISEASE IN-BAND PRECISION (N={len(test)} held-out, gold=MONDO id) ===")
print(f"  {'band':>12} | {'n':>6} | {'precision':>9} | false-ground")
bands = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.0, 0.80)]
sims_all = np.empty(len(test)); pred = np.empty(len(test), dtype=object)
for s in range(0, len(test), 2048):
    M = Tq[s:s+2048] @ I.T; j = M.argmax(1); sims_all[s:s+2048] = M.max(1)
    for k, jj in enumerate(j): pred[s+k] = tn_mid[jj]
for lo, hi in bands:
    m = (sims_all >= lo) & (sims_all < hi); n = int(m.sum())
    if n: corr = int((pred[m] == gold[m]).sum()); print(f"  [{lo:.2f},{hi:.2f}) | {n:6d} | {corr/n:9.3f} | {1-corr/n:.1%}")

# (B) NIL-REJECTION (hold out ONE WHOLE concept per query -> that concept ABSENT, rest of registry intact;
# false-ground = the query still resolves to a (wrong) DIFFERENT concept at sim>=thr -> should have abstained)
from collections import defaultdict
mid2rows = defaultdict(list)
for r, m in enumerate(tn_mid): mid2rows[m].append(r)
sample_mids = list(dict.fromkeys(tn_mid[list(range(0, len(tnorms), max(1, len(tnorms)//4000)))[:4000]]))
print(f"\n=== (B) NIL-REJECTION ({len(sample_mids)} held-out-WHOLE concepts; true answer ABSENT -> should abstain) ===")
fg = {t: 0 for t in (0.85, 0.90, 0.92, 0.95)}
for m in sample_mids:
    rows = mid2rows[m]; i = rows[0]              # query = first form of this concept
    sims = (I[i] @ I.T).copy(); sims[rows] = -1.0   # mask ONLY this concept's forms (rest of registry stays)
    s = float(sims.max())
    for t in fg:
        if s >= t: fg[t] += 1
n = len(sample_mids)
print(f"  {'abstain thr':>11} | {'false-ground':>12} | correct-abstain")
for t in sorted(fg):
    print(f"  {t:>11.2f} | {fg[t]/n:12.1%} | {1-fg[t]/n:.1%}")
print("\n  -> false-ground = resolver confidently grounds an ABSENT concept to a wrong neighbor (the dangerous error).")
print("     Higher abstain threshold lowers false-ground but also lowers recall — this is the operating-point / reranker tradeoff.")
