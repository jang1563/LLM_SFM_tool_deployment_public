"""Reranker prototype (validate before HPC) — does a closed-set LLM rerank fix the 0.85-0.95 sibling-drift?
For held-out disease forms in that band (SapBERT top-1 only 50-66% precise), take SapBERT's top-8 DISTINCT
candidate MONDO concepts and ask Claude to pick the one that is the SAME disease (or 0 = none = abstain).
Measures: SapBERT top-1 precision (baseline) vs reranked precision, and recall@8 (the ceiling the reranker
can reach). Non-circular (gold=MONDO id). Sampled to keep it cheap; the deployable is a closed-set Qwen3-8B."""
import re, os, json, numpy as np
from paths import A2_DIR
A2 = str(A2_DIR)
GENERIC = {"disease","diseases","disorder","disorders","syndrome","syndromes","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication"}
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

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
names = json.load(open(A2 + "/mondo_hierarchy.json"))["names"]
test = []; train = {}; held_set = set()
for mid, forms in id2forms.items():
    fs = sorted(forms)
    if len(fs) >= 3: held = fs[len(fs)//2]; held_set.add((held, mid)); test.append((held, mid))
for mid, forms in id2forms.items():
    for f in forms:
        if (f, mid) in held_set: continue
        train.setdefault(f, mid)
tnorms = list(train); tn_mid = np.array([train[t] for t in tnorms])
I = np.load(A2 + "/disease_index.npy"); Tq = np.load(A2 + "/disease_test.npy")

# pick the 0.85-0.95 band (the low-precision zone), deterministic sample
band = []
for k, (held, mid) in enumerate(test):
    s = float((Tq[k] @ I.T).max())
    if 0.85 <= s < 0.95: band.append(k)
SAMPLE = band[::max(1, len(band)//120)][:120]
print(f"0.85-0.95 band: {len(band)} forms; sampling {len(SAMPLE)}", flush=True)

import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
import sys as _sys
MODEL = _sys.argv[1] if len(_sys.argv) > 1 else "claude-haiku-4-5"

sap_correct = rerank_correct = gold_in_cands = abstain = 0
for ki, k in enumerate(SAMPLE):
    held, gold = test[k]
    sims = Tq[k] @ I.T; order = np.argsort(-sims)
    cands = []; seen = set()                                  # top-8 DISTINCT concepts
    for j in order:
        mid = tn_mid[j]
        if mid in seen: continue
        seen.add(mid); cands.append(mid)
        if len(cands) >= 8: break
    sap_top1 = cands[0]
    if sap_top1 == gold: sap_correct += 1
    if gold in cands: gold_in_cands += 1
    lines = "\n".join(f"{i+1}. {names.get(c, c)}" for i, c in enumerate(cands))
    prompt = (f'A clinical-trial condition string must be matched to ONE disease-ontology concept.\n'
              f'Condition string: "{held}"\n\nCandidate concepts:\n{lines}\n\n'
              f'Which candidate is the SAME disease as the condition string? '
              f'Reply with ONLY the number (1-{len(cands)}), or 0 if none is the same disease.')
    try:
        r = client.messages.create(model=MODEL, max_tokens=8, messages=[{"role": "user", "content": prompt}])
        m = re.search(r"\d+", r.content[0].text); pick = int(m.group()) if m else 0
    except Exception as e:
        pick = -1
    if pick == 0: abstain += 1
    elif 1 <= pick <= len(cands) and cands[pick-1] == gold: rerank_correct += 1
    if (ki + 1) % 30 == 0: print(f"  {ki+1}/{len(SAMPLE)}", flush=True)
N = len(SAMPLE)
print(f"\n=== RERANKER PROTOTYPE (0.85-0.95 band, N={N}, model={MODEL}) ===")
print(f"  gold-in-top8 (ceiling)   : {gold_in_cands/N:.3f}")
print(f"  SapBERT top-1 precision  : {sap_correct/N:.3f}  (baseline)")
print(f"  LLM-rerank precision     : {rerank_correct/N:.3f}  ({'+' if rerank_correct>=sap_correct else ''}{(rerank_correct-sap_correct)/N:+.0%})")
print(f"  LLM abstained (said None): {abstain/N:.1%}")
print(f"  -> rerank precision among gold-in-top8: {rerank_correct/max(gold_in_cands,1):.3f}")
