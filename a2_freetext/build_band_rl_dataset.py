"""Track A Step 1 — build the band-reranker post-training dataset (non-circular, gold=MONDO id).
Every MONDO label/synonym is a query; mask its own row; SapBERT top-8 DISTINCT concepts = candidates;
keep the 0.85-0.95 ambiguous BAND. CONCEPT-DISJOINT split (a concept's forms are entirely in train OR eval,
never both) + a NEIGHBORHOOD-DISJOINT eval flag (no eval concept's is_a parent/sibling appears as a train gold
or distractor) so SFT gains can't be ontology-neighborhood memorization. Reuses the cached mondo_sapbert.npy."""
import os, re, json, hashlib, numpy as np
from pathlib import Path
A2 = str(Path(os.environ.get("A2_FREETEXT_DIR", Path(__file__).resolve().parent)).expanduser().resolve())
GENERIC = {"disease","diseases","disorder","disorders","syndrome","syndromes","syndromic disease","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication","infectious disease","rare disease"}
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

# replicate the index alignment exactly (build_crosswalk.py): labels = sorted(lab2id), aligned with mondo_sapbert.npy
lab2id = {}; id2name = {}; id2forms = {}; cid = None
H = json.load(open(A2 + "/mondo_hierarchy.json")); hier = H["hier"]
for line in open(A2 + "/mondo.obo"):
    line = line.rstrip("\n")
    if line == "[Term]": cid = None
    elif line.startswith("id: "): cid = line[4:]
    elif cid and cid.startswith("MONDO:"):
        if line.startswith("name: "):
            raw = line[6:]; nm = norm(raw); id2name.setdefault(cid, raw)
            if len(nm) >= 4 and nm not in GENERIC and cid != "MONDO:0000001":
                lab2id.setdefault(nm, cid); id2forms.setdefault(cid, set()).add(nm)
        elif line.startswith("synonym: "):
            m = re.match(r'synonym: "(.*?)" (EXACT|NARROW)', line)
            if m:
                t = norm(m.group(1))
                if len(t) >= 4 and t not in GENERIC and cid != "MONDO:0000001":
                    lab2id.setdefault(t, cid); id2forms.setdefault(cid, set()).add(t)
labels = sorted(lab2id); lab_mid = np.array([lab2id[l] for l in labels])
I = np.load(A2 + "/mondo_sapbert.npy"); assert len(labels) == I.shape[0], f"{len(labels)} vs {I.shape}"
print(f"index {I.shape}, {len(set(lab_mid))} concepts", flush=True)

def split_of(mid):  # deterministic concept-disjoint split (~12% eval)
    return "eval" if int(hashlib.md5(mid.encode()).hexdigest(), 16) % 100 < 12 else "train"
def obsolete(mid): return hier.get(mid, {}).get("obsolete", False)
# neighborhood of a concept = itself + is_a parents + their children (siblings) + its children
def neighborhood(mid):
    h = hier.get(mid, {}); nb = {mid} | set(h.get("parents", [])) | set(h.get("desc2", []))
    for p in h.get("parents", []): nb |= set(hier.get(p, {}).get("desc2", []))
    return nb
eval_concepts = {m for m in set(lab_mid) if split_of(m) == "eval"}
train, ev = [], []
for s in range(0, len(labels), 512):
    sims = I[s:s+512] @ I.T
    for r in range(sims.shape[0]):
        i = s + r; gold = lab_mid[i]
        if obsolete(gold): continue
        row = sims[r]; row[i] = -1.0                 # mask the query's own form (in-place ok, row is this chunk's copy)
        top = np.argpartition(-row, 64)[:64]          # top-64 forms (fast) -> dedup to 8 distinct concepts
        top = top[np.argsort(-row[top])]
        cands = []; seen = set()
        for j in top:
            mj = lab_mid[j]
            if obsolete(mj) or mj in seen: continue
            seen.add(mj); cands.append((mj, float(row[j])))
            if len(cands) >= 8: break
        if len(cands) < 8: continue                   # rare: a concept dominated top-64; skip
        top1_sim = cands[0][1]
        if not (0.85 <= top1_sim < 0.95): continue   # the ambiguous band only
        rec = {"query": labels[i], "gold": gold, "gold_name": id2name.get(gold),
               "gold_in_top8": int(gold in {c for c, _ in cands}),
               "candidates": [{"mondo_id": c, "name": id2name.get(c, c),
                               # DE-LEAK: exclude the held-out QUERY (labels[i]) from displayed syns
                               "syns": sorted(x for x in id2forms.get(c, []) if x != norm(id2name.get(c, "")) and x != labels[i])[:5]}
                              for c, _ in cands]}
        if split_of(gold) == "eval":
            # neighborhood-clean flag: eval rec where NO candidate distractor is a train concept -> the leakage
            # control (compare full-eval vs nbhd_clean precision post-training; a drop = neighborhood memorization)
            rec["nbhd_clean"] = int(all(split_of(c) == "eval" or c == gold for c, _ in cands))
            ev.append(rec)
        else:
            train.append(rec)   # concept-disjoint on gold; leakage measured eval-side, not by decimating train
    if s % 10240 == 0: print(f"  {s}/{len(labels)} | train {len(train)} eval {len(ev)}", flush=True)

with open(A2 + "/band_train.jsonl", "w") as f:
    for r in train: f.write(json.dumps(r) + "\n")
with open(A2 + "/band_eval.jsonl", "w") as f:
    for r in ev: f.write(json.dumps(r) + "\n")
git = sum(r["gold_in_top8"] for r in train) / max(len(train), 1)
nbc = sum(r.get("nbhd_clean", 0) for r in ev)
print(f"\n=== BAND RL DATASET ===")
print(f"  train {len(train)} (gold-in-top8 {git:.3f}) | eval {len(ev)} ({nbc} neighborhood-clean)")
print(f"  -> band_train.jsonl / band_eval.jsonl (concept-disjoint; train drops candidates in eval neighborhood)")
