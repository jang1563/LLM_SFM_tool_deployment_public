"""Disease-leg backbone bake-off: SapBERT vs BioLORD-2023-C on the SAME MONDO LOO (gold=MONDO id).
recall@1 = nearest index form is the gold concept (no threshold -> isolates ranking/backbone ability;
also captures sibling-drift, since landing on a sibling concept counts as a miss). SapBERT reuses the
cached CLS embeddings; BioLORD embeds fresh with MEAN pooling (sentence-transformers convention)."""
import re, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from paths import A2_DIR
A2 = str(A2_DIR)
GENERIC = {"disease","diseases","disorder","disorders","syndrome","syndromes","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication"}
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

# rebuild the IDENTICAL test/train split (deterministic)
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
tnorms = list(train); N = len(test)
print(f"test {N} | index {len(tnorms)}", flush=True)
gold = np.array([mid for _, mid in test])
tn_mid = np.array([train[t] for t in tnorms])

def recall1(I, Tq, chunk=2048):
    hit = 0
    for s in range(0, len(test), chunk):
        sims = Tq[s:s+chunk] @ I.T; j = sims.argmax(1)
        hit += int((tn_mid[j] == gold[s:s+chunk]).sum())
    return hit / N

dev = "mps"
# --- SapBERT (cached CLS) ---
Is = np.load(A2 + "/disease_index.npy"); Ts = np.load(A2 + "/disease_test.npy")
print(f"SapBERT  recall@1: {recall1(Is, Ts):.3f}", flush=True)

# --- BioLORD-2023-C (mean pooling) ---
M = "FremyCompany/BioLORD-2023-C"
tok = AutoTokenizer.from_pretrained(M); model = AutoModel.from_pretrained(M).to(dev).eval()
def embed_mean(strs, bs=384):
    out = []
    for i in range(0, len(strs), bs):
        t = tok(strs[i:i+bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        with torch.no_grad(): h = model(**t).last_hidden_state
        mask = t.attention_mask.unsqueeze(-1).float()
        e = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
        if i % 25600 == 0: print(f"  biolord embed {i}/{len(strs)}", flush=True)
    return torch.cat(out).numpy().astype("float32")
print("embedding index (BioLORD)...", flush=True); Ib = embed_mean(tnorms)
print("embedding test (BioLORD)...", flush=True); Tb = embed_mean([h for h, _ in test])
print(f"BioLORD  recall@1: {recall1(Ib, Tb):.3f}", flush=True)
print("\n=== DISEASE BACKBONE BAKE-OFF (MONDO LOO, recall@1, N=%d) ===" % N)
print(f"  SapBERT : {recall1(Is, Ts):.3f}")
print(f"  BioLORD : {recall1(Ib, Tb):.3f}")
