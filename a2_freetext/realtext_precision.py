"""Hyper-review fix for the 'proxy not real free-text' caveat: judge the resolver's resolutions on REAL
clinicaltrials.gov condition strings with an LLM judge (gold = LLM judgement, not MONDO's own synonym
grouping). Samples the materialized crosswalk stratified by tier (dict / sapbert>=0.95 / 0.90-0.95 / 0.85-0.90)
and asks sonnet whether each condition->MONDO mapping is CORRECT/BROADER/NARROWER/WRONG -> real-input precision."""
import os, json, re, random
from paths import A2_DIR
A2 = str(A2_DIR)
cx = json.load(open(A2 + "/conditions_mondo_crosswalk.json"))
rows = list(cx.values())
def tier(v):
    if v["match"] == "dict": return "dict"
    c = v.get("conf") or 0
    return "sapbert>=0.95" if c >= 0.95 else ("sapbert 0.90-0.95" if c >= 0.90 else "sapbert 0.85-0.90")
buckets = {"dict": [], "sapbert>=0.95": [], "sapbert 0.90-0.95": [], "sapbert 0.85-0.90": []}
for v in rows: buckets[tier(v)].append(v)
rng = random.Random(42)
SAMPLE = []
for t, n in [("dict", 60), ("sapbert>=0.95", 60), ("sapbert 0.90-0.95", 50), ("sapbert 0.85-0.90", 40)]:
    b = buckets[t]; rng.shuffle(b); SAMPLE += [(t, v) for v in b[:n]]
print(f"sampled {len(SAMPLE)} real conditions across tiers", flush=True)

import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"
from collections import defaultdict
res = defaultdict(lambda: defaultdict(int))
for i, (t, v) in enumerate(SAMPLE):
    prompt = (f'A clinicaltrials.gov condition string was mapped to a disease-ontology (MONDO) concept.\n'
              f'Condition string: "{v["condition_name"]}"\n'
              f'Mapped to concept: "{v["mondo_name"]}"\n\n'
              f'Judge the mapping. Reply ONE word:\n'
              f'CORRECT = same disease; BROADER = mapped concept is a more general parent; '
              f'NARROWER = mapped concept is a subtype of the condition; WRONG = different disease.')
    try:
        r = client.messages.create(model=MODEL, max_tokens=8, messages=[{"role": "user", "content": prompt}])
        ans = r.content[0].text.strip().upper()
        verdict = next((w for w in ("CORRECT", "BROADER", "NARROWER", "WRONG") if w in ans), "WRONG")
    except Exception:
        verdict = "ERR"
    res[t][verdict] += 1
    if (i + 1) % 50 == 0: print(f"  {i+1}/{len(SAMPLE)}", flush=True)

print(f"\n=== REAL-INPUT PRECISION (LLM-judged, model={MODEL}) ===")
print(f"  {'tier':>18} | {'n':>3} | {'CORRECT':>7} | {'+hier(B/N)':>10} | WRONG")
for t in ("dict", "sapbert>=0.95", "sapbert 0.90-0.95", "sapbert 0.85-0.90"):
    d = res[t]; n = sum(d.values())
    if not n: continue
    corr = d["CORRECT"]; hier = d["BROADER"] + d["NARROWER"]; wrong = d["WRONG"] + d["ERR"]
    print(f"  {t:>18} | {n:>3} | {corr/n:7.2f} | {(corr+hier)/n:10.2f} | {wrong/n:.2f}")
print("  (CORRECT = exact same disease; +hier counts parent/subtype as usable grounding; WRONG = false-ground)")
