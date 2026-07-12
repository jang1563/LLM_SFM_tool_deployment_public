"""Zero-API reranker proof: run a LOCAL OPEN model (Qwen2.5-7B-Instruct, BeLink Qwen3-8B class) on MPS as the
closed-set band reranker, vs the Claude-haiku reference (0.900 de-leaked). The corrected deployment thesis is
that a local open stack closes most of the gap, not that it exactly matches the proprietary reference.
Reads precomputed band_cases.json (no SapBERT in memory)."""
import re, os, sys, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
A2 = str(Path(os.environ.get("A2_FREETEXT_DIR", Path(__file__).resolve().parent)).expanduser().resolve())
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
cases = json.load(open(A2 + "/band_cases.json"))
N_CASES = int(sys.argv[2]) if len(sys.argv) > 2 else 50   # MPS 7B ~15s/case; 50 is enough for the +20pp signal
cases = cases[:N_CASES]
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
print(f"{MODEL} loaded on {dev}; {len(cases)} band cases", flush=True)

def pick(query, cands):
    lines = []
    for i, c in enumerate(cands):
        lines.append(f"{i+1}. {c['name']}" + (f"  [aka: {'; '.join(c['syns'])}]" if c["syns"] else ""))
    prompt = (f'Match a clinical-trial condition string to ONE disease-ontology concept.\n'
              f'Condition string: "{query}"\n\nCandidates (with synonyms):\n' + "\n".join(lines) +
              f'\n\nWhich candidate is the SAME disease? Reply ONLY the number (1-{len(cands)}), or 0 if none.')
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\d+", ans); return int(m.group()) if m else 0

sap = rer = gold_in = 0
for n, c in enumerate(cases):
    sap += c["sap_correct"]; gold_in += c["gold_in"]
    cands = c["candidates"]; p = pick(c["query"], cands)
    if 1 <= p <= len(cands) and cands[p-1]["mondo_id"] == c["gold"]: rer += 1
    if (n + 1) % 50 == 0: print(f"  {n+1}/{len(cases)}", flush=True)
N = len(cases)
print(f"\n=== LOCAL OPEN-MODEL RERANKER ({MODEL}, N={N}) ===")
print(f"  gold-in-top8 ceiling : {gold_in/N:.3f}")
print(f"  SapBERT top-1        : {sap/N:.3f}")
print(f"  {MODEL.split('/')[-1]:<22}: {rer/N:.3f}   ({(rer-sap)/N:+.0%} vs SapBERT)")
print("  (Corrected Claude-haiku reference: 0.900 de-leaked; open-model deployment is valuable if it closes most of this gap)")
