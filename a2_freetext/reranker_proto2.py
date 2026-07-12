"""Hyper-review fix for the reranker verdict: N=400 (not 120), candidates shown with their FULL synonym set
(not canonical-name-only, which under-fed the LLM in ~76% of cases), and a PAIRED McNemar test + Wald CIs so
'saturated' is supported or rejected with statistics. Settles whether the +7-8pp gain and the 0.867 ceiling
were real or artifacts of small-N / name-starvation."""
import re, os, json, math, numpy as np
from paths import A2_DIR
A2 = str(A2_DIR)
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
band = [k for k in range(len(test)) if 0.85 <= float((Tq[k] @ I.T).max()) < 0.95]
SAMPLE = band[:400]
print(f"band {len(band)}; sampling {len(SAMPLE)}", flush=True)

import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-haiku-4-5"
sap_v = []; rer_v = []; gold_in = 0
for ki, k in enumerate(SAMPLE):
    held, gold = test[k]
    order = np.argsort(-(Tq[k] @ I.T)); cands = []; seen = set()
    for j in order:
        m = tn_mid[j]
        if m in seen: continue
        seen.add(m); cands.append(m)
        if len(cands) >= 8: break
    sap_v.append(1 if cands[0] == gold else 0); gold_in += (gold in cands)
    # show each candidate with its FULL synonym set (the fix)
    lines = []
    for i, c in enumerate(cands):
        syns = sorted(s for s in id2forms.get(c, []) if s != norm(names.get(c, "")))[:5]
        label = names.get(c, c) + (f"  [aka: {'; '.join(syns)}]" if syns else "")
        lines.append(f"{i+1}. {label}")
    prompt = (f'Match a clinical-trial condition string to ONE disease-ontology concept.\n'
              f'Condition string: "{held}"\n\nCandidates (with synonyms):\n' + "\n".join(lines) +
              f'\n\nWhich candidate is the SAME disease? Reply ONLY the number (1-{len(cands)}), or 0 if none.')
    try:
        r = client.messages.create(model=MODEL, max_tokens=8, messages=[{"role": "user", "content": prompt}])
        mm = re.search(r"\d+", r.content[0].text); pick = int(mm.group()) if mm else 0
    except Exception:
        pick = -1
    rer_v.append(1 if (1 <= pick <= len(cands) and cands[pick-1] == gold) else 0)
    if (ki + 1) % 80 == 0: print(f"  {ki+1}/{len(SAMPLE)}", flush=True)

sap_v = np.array(sap_v); rer_v = np.array(rer_v); N = len(sap_v)
def ci(p, n): h = 1.96 * math.sqrt(p * (1 - p) / n); return f"[{max(0,p-h):.3f}-{min(1,p+h):.3f}]"
b = int(((sap_v == 1) & (rer_v == 0)).sum()); c = int(((sap_v == 0) & (rer_v == 1)).sum())
mcnemar = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
print(f"\n=== RERANKER N={N}, full-synonym candidates, model={MODEL} ===")
print(f"  gold-in-top8 ceiling : {gold_in/N:.3f} {ci(gold_in/N,N)}")
print(f"  SapBERT top-1        : {sap_v.mean():.3f} {ci(sap_v.mean(),N)}")
print(f"  LLM-rerank           : {rer_v.mean():.3f} {ci(rer_v.mean(),N)}   ({(rer_v.mean()-sap_v.mean()):+.3f})")
print(f"  McNemar b={b} (sap-only-right) c={c} (rerank-only-right) chi2={mcnemar:.2f} {'SIGNIFICANT' if mcnemar>3.84 else 'ns (p>0.05)'}")
print(f"  -> rerank among gold-in-top8: {(rer_v.sum()/max(gold_in,1)):.3f}")
