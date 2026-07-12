"""Dump the 0.85-0.95 band cases (query + SapBERT top-8 candidates with synonyms + gold) to JSON, using
cached embeddings only (no model). Decouples candidate generation from reranking so the local open-model
reranker can run without SapBERT in memory. N=200 deterministic sample."""
import os, re, json, numpy as np
from pathlib import Path
A2 = str(Path(os.environ.get("A2_FREETEXT_DIR", Path(__file__).resolve().parent)).expanduser().resolve())
GEN = {"disease","diseases","disorder","disorders","syndrome","syndromes","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication"}
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()

id2forms = {}; cur = None
for line in open(A2 + "/mondo.obo"):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: "): cur = line[4:]
    elif cur and cur.startswith("MONDO:"):
        if line.startswith("is_obsolete: true"): id2forms.pop(cur, None); cur = None
        elif line.startswith("name: "):
            n = norm(line[6:])
            if len(n) >= 4 and n not in GEN: id2forms.setdefault(cur, set()).add(n)
        elif line.startswith("synonym: "):
            m = re.match(r'synonym: "(.*?)" (EXACT|NARROW)', line)
            if m:
                n = norm(m.group(1))
                if len(n) >= 4 and n not in GEN: id2forms.setdefault(cur, set()).add(n)
names = json.load(open(A2 + "/mondo_hierarchy.json"))["names"]
test = []; train = {}; held = set()
for mid, forms in id2forms.items():
    fs = sorted(forms)
    if len(fs) >= 3: h = fs[len(fs)//2]; held.add((h, mid)); test.append((h, mid))
for mid, forms in id2forms.items():
    for f in forms:
        if (f, mid) in held: continue
        train.setdefault(f, mid)
tn = list(train); tnmid = np.array([train[t] for t in tn])
I = np.load(A2 + "/disease_index.npy"); Tq = np.load(A2 + "/disease_test.npy")
band = [k for k in range(len(test)) if 0.85 <= float((Tq[k] @ I.T).max()) < 0.95]
SAMPLE = band[:200]
cases = []
for k in SAMPLE:
    held_q, gold = test[k]
    order = np.argsort(-(Tq[k] @ I.T)); cands = []; seen = set()
    for j in order:
        m = tnmid[j]
        if m in seen: continue
        seen.add(m); cands.append(m)
        if len(cands) >= 8: break
    cases.append({"query": held_q, "gold": gold, "gold_name": names.get(gold),
                  "sap_top1": cands[0], "sap_correct": int(cands[0] == gold), "gold_in": int(gold in cands),
                  "candidates": [{"mondo_id": c, "name": names.get(c, c),
                                  # DE-LEAK: exclude the held-out QUERY from displayed syns (it re-entered the
                                  # gold candidate's aka-list verbatim -> a no-model string-match scored ~0.92)
                                  "syns": sorted(s for s in id2forms.get(c, []) if s != norm(names.get(c, "")) and s != held_q)[:5]}
                                 for c in cands]})
json.dump(cases, open(A2 + "/band_cases.json", "w"))
sc = sum(c["sap_correct"] for c in cases); gi = sum(c["gold_in"] for c in cases)
print(f"band_cases.json: {len(cases)} cases | SapBERT top-1 {sc/len(cases):.3f} | gold-in-top8 {gi/len(cases):.3f}")
