"""Hyper-review fix: re-run BOTH LOO legs with a RANDOM held form (seed=42) instead of the middle of the
alphabetical sort. The middle-of-sorted holdout keeps a same-prefix sibling in the index ~71% vs ~59% random,
inflating recall ~+10pp. Same norm/method as the originals — isolates the holdout-SELECTION effect.
Reports the honest absolute recall ladder; the SapBERT-vs-BioLORD RELATIVE comparison is unaffected (shared split)."""
import re, sqlite3, json, random, numpy as np, torch
from collections import defaultdict
from rapidfuzz import fuzz, process
from transformers import AutoTokenizer, AutoModel
from paths import A2_DIR, CT_DB
A2 = str(A2_DIR)
CT = str(CT_DB)
DOSE = re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|ug|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w*|part\s*\d+|cohort\s*\w+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|sodium|hydrochloride|hcl|sulfate|mesylate|bitartrate|acetate|injection|capsules?|tablet|for inhalation|q\dw|once weekly|bid|qd)\b", re.I)
def dnorm(s):  # drug norm (matches original eval_drug_loo)
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = DOSE.sub(" ", s); return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()
GEN = {"disease","diseases","disorder","disorders","syndrome","syndromes","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication"}
def cnorm(s):  # disease norm
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()

dev = "mps"; M = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tok = AutoTokenizer.from_pretrained(M); model = AutoModel.from_pretrained(M).to(dev).eval()
def embed(strs, bs=384):
    out = []
    for i in range(0, len(strs), bs):
        t = tok(strs[i:i+bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        with torch.no_grad(): e = model(**t).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
        if i % 25600 == 0 and len(strs) > 5000: print(f"  embed {i}/{len(strs)}", flush=True)
    return torch.cat(out).numpy().astype("float32")

# ---------- DRUG ----------
alias = json.load(open(A2 + "/chembl_aliases.json"))
con = sqlite3.connect(CT)
fail = set(r[0] for r in con.execute("SELECT DISTINCT intervention_id FROM trial_failure_results"))
ch2raw = defaultdict(set)
for iid, nm, ch in con.execute("SELECT intervention_id, intervention_name, chembl_id FROM interventions WHERE chembl_id IS NOT NULL AND intervention_name IS NOT NULL"):
    if iid in fail and nm.strip(): ch2raw[ch].add(nm.strip())
rng = random.Random(42)
test = []; train_raw = {}; held_set = set()
for ch, raws in ch2raw.items():
    rs = sorted(raws)
    if len(rs) >= 3: held = rng.choice(rs); held_set.add((held, ch)); test.append((held, ch))
for ch, raws in ch2raw.items():
    for r in raws:
        if (r, ch) in held_set: continue
        n = dnorm(r)
        if n: train_raw.setdefault(n, ch)
tn = list(train_raw); Nd = len(test)
print(f"DRUG random-holdout: test {Nd} | index {len(tn)} — embedding...", flush=True)
I = embed(tn); Tq = embed([dnorm(h) for h, _ in test]); tnch = [train_raw[t] for t in tn]
nmh = kbh = uni = 0
for k, (held, ch) in enumerate(test):
    n = dnorm(held); a = train_raw.get(n) == ch; b = alias.get(n) == ch
    c = float((Tq[k] @ I.T).max()) >= 0.92 and tnch[int((Tq[k] @ I.T).argmax())] == ch
    nmh += a; kbh += (a or b); uni += (a or b or c)
print(f"\n=== DRUG LOO RANDOM-HOLDOUT (gold=chembl, N={Nd}) ===")
print(f"  +dosage-normalize : {nmh/Nd:.3f}   (middle-form was 0.533)")
print(f"  +ChEMBL-KB        : {kbh/Nd:.3f}   (middle-form was 0.690)")
print(f"  RESOLVER (union)  : {uni/Nd:.3f}   (middle-form was 0.739)")

# ---------- DISEASE ----------
id2forms = {}; cur = None
for line in open(A2 + "/mondo.obo"):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: "): cur = line[4:]
    elif cur and cur.startswith("MONDO:"):
        if line.startswith("is_obsolete: true"): id2forms.pop(cur, None); cur = None
        elif line.startswith("name: "):
            n = cnorm(line[6:])
            if len(n) >= 4 and n not in GEN: id2forms.setdefault(cur, set()).add(n)
        elif line.startswith("synonym: "):
            m = re.match(r'synonym: "(.*?)" (EXACT|NARROW)', line)
            if m:
                n = cnorm(m.group(1))
                if len(n) >= 4 and n not in GEN: id2forms.setdefault(cur, set()).add(n)
rng2 = random.Random(42)
dtest = []; dtrain = {}; dheld = set()
for mid, forms in id2forms.items():
    fs = sorted(forms)
    if len(fs) >= 3: held = rng2.choice(fs); dheld.add((held, mid)); dtest.append((held, mid))
for mid, forms in id2forms.items():
    for f in forms:
        if (f, mid) in dheld: continue
        dtrain.setdefault(f, mid)
dtn = list(dtrain); Ndis = len(dtest); dtnmid = np.array([dtrain[t] for t in dtn]); dgold = np.array([m for _, m in dtest])
print(f"\nDISEASE random-holdout: test {Ndis} | index {len(dtn)} — embedding...", flush=True)
DI = embed(dtn); DTq = embed([h for h, _ in dtest])
r1 = thr = fz = 0; fzn = 0
for k in range(Ndis):
    sims = DTq[k] @ DI.T; j = int(sims.argmax()); s = float(sims[j])
    if dtnmid[j] == dgold[k]:
        r1 += 1
        if s >= 0.90: thr += 1
    if k < 1500:
        fzn += 1; m = process.extractOne(dtest[k][0], dtn, scorer=fuzz.token_sort_ratio)
        if m and m[1] >= 90 and dtrain[m[0]] == dgold[k]: fz += 1
print(f"\n=== DISEASE LOO RANDOM-HOLDOUT (gold=MONDO id, N={Ndis}) ===")
print(f"  fuzzy(>=90)      : {fz/fzn:.3f}  (sampled n={fzn}; middle-form was 0.668)")
print(f"  SapBERT recall@1 : {r1/Ndis:.3f}   (middle-form was 0.766)")
print(f"  SapBERT thr>=0.90: {thr/Ndis:.3f}   (middle-form was 0.682)")
