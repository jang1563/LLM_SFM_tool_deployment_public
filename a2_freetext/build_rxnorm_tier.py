"""RxNorm drug tier (Task #1, UMLS approved 2026-06-28) — recover the drug RESIDUE the deterministic tiers miss.
Residue = failure-bearing drug names NOT in the CT registry chembl index AND NOT in the ChEMBL-alias KB AND not
vague/combo. For each: public RxNav API name -> RXCUI -> ingredient (brand->generic), then bridge ingredient ->
chembl via chembl_aliases.json. Measures the recovery rate on a sample, writes rxnorm_tier.json (name->chembl)."""
import re, sqlite3, json, sys, time, urllib.request, urllib.parse
from paths import A2_DIR, CT_DB
A2 = str(A2_DIR)
CT = str(CT_DB)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
DOSAGE = re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|ug|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w*|part\s*\d+|cohort\s*\w+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|injection|capsules?|tablet|for inhalation|q\dw|once weekly|bid|qd)\b", re.I)
SALT = re.compile(r"\b(sodium|hydrochloride|hcl|sulfate|sulphate|mesylate|bitartrate|tartrate|acetate|citrate|phosphate|maleate|fumarate|succinate|hydrobromide|besylate|tosylate)\b", re.I)
CATION = {"magnesium","calcium","sodium","potassium","zinc","iron","ferrous","ferric","lithium","aluminum","aluminium","ammonium","chloride","bicarbonate","carbonate","selenium","copper","manganese","strontium"}
VAGUE = re.compile(r"placebo|comparator|standard of care|best supportive|investigational|study drug|matching|vehicle|sham|\+|/| and |combination|regimen|background therapy", re.I)
def _clean(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower()); base = _clean(DOSAGE.sub(" ", s)); salted = _clean(SALT.sub(" ", base))
    return base if (not salted or salted in CATION) else salted

alias = json.load(open(A2 + "/chembl_aliases.json"))
con = sqlite3.connect(CT)
fail = set(r[0] for r in con.execute("SELECT DISTINCT intervention_id FROM trial_failure_results"))
name2ch = {}; raw_names = {}
for iid, nm, ch in con.execute("SELECT intervention_id, intervention_name, chembl_id, intervention_type FROM interventions".replace(", intervention_type","")+" WHERE intervention_name IS NOT NULL"):
    pass
residue = {}
for iid, nm, ch, ty in con.execute("SELECT intervention_id, intervention_name, chembl_id, intervention_type FROM interventions WHERE intervention_name IS NOT NULL"):
    if ch: name2ch[norm(nm)] = ch
for iid, nm, ty in con.execute("SELECT intervention_id, intervention_name, intervention_type FROM interventions WHERE intervention_name IS NOT NULL"):
    if iid not in fail: continue
    if ty and ty.lower() not in ("drug", "biological", "biologic"): continue
    n = norm(nm)
    if not n or n in name2ch or n in alias or VAGUE.search(nm): continue
    residue.setdefault(n, nm)
res = sorted(residue)
print(f"drug residue (failure-bearing, not in registry/KB/vague): {len(res)} unique names; RxNav sampling {min(N,len(res))}", flush=True)

def rxnav(path):
    try:
        with urllib.request.urlopen("https://rxnav.nlm.nih.gov/REST/" + path, timeout=8) as r: return json.load(r)
    except Exception: return {}
def ingredient(name):
    d = rxnav(f"rxcui.json?name={urllib.parse.quote(name)}")
    ids = d.get("idGroup", {}).get("rxnormId", [])
    if not ids:
        d = rxnav(f"rxcui.json?name={urllib.parse.quote(name)}&search=2")  # approximate
        ids = d.get("idGroup", {}).get("rxnormId", [])
    if not ids: return None
    d = rxnav(f"rxcui/{ids[0]}/related.json?tty=IN")
    for g in d.get("relatedGroup", {}).get("conceptGroup", []):
        if g.get("conceptProperties"): return g["conceptProperties"][0]["name"]
    return None

tier = {}; rx_hit = bridged = 0
for i, n in enumerate(res[:N]):
    ing = ingredient(n)
    if ing:
        rx_hit += 1; ni = norm(ing); ch = alias.get(ni)
        if ch: tier[n] = ch; bridged += 1
    if (i + 1) % 50 == 0: print(f"  {i+1}/{min(N,len(res))} | rxnav-hit {rx_hit} bridged->chembl {bridged}", flush=True)
    time.sleep(0.05)
json.dump(tier, open(A2 + "/rxnorm_tier.json", "w"))
n_s = min(N, len(res))
print(f"\n=== RxNorm TIER (sample {n_s}) ===")
print(f"  RxNav ingredient resolved : {rx_hit}/{n_s} = {rx_hit/n_s:.1%}")
print(f"  bridged ingredient->chembl: {bridged}/{n_s} = {bridged/n_s:.1%}  -> rxnorm_tier.json ({len(tier)} name->chembl)")
print(f"  => the RxNorm tier recovers ~{bridged/n_s:.0%} of the previously-abstained drug residue ({len(res)} total)")
