"""Build the curated ChEMBL synonym KB: normalized drug name/alias -> chembl_id, from chembl_37
molecule_synonyms (RESEARCH_CODE/TRADE_NAME/INN/USAN/FDA...) + molecule_dictionary.pref_name.
This is the #1 lever for the drug leg (the LOO eval showed SapBERT can't bridge brand<->generic;
it's arbitrary naming -> a curated alias table is the right tool). Output: chembl_aliases.json."""
import sqlite3, json, re
from collections import Counter
from paths import A2_DIR, CHEMBL_DB
DB = str(CHEMBL_DB)
A2 = str(A2_DIR)
# MUST stay in sync with resolve_drug.py: DOSAGE always stripped, SALT stripped only if it doesn't leave a bare cation.
DOSAGE = re.compile(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|ug|g|ml|iu|units?|%|ppm)|formulation\s*\w+|dose\s*\w*|part\s*\d+|cohort\s*\w+|open label|monotherapy|intravenous|oral|topical|inhal\w*|subcutaneous|injection|capsules?|tablet|for inhalation|q\dw|once weekly|bid|qd)\b", re.I)
SALT = re.compile(r"\b(sodium|hydrochloride|hcl|sulfate|sulphate|mesylate|bitartrate|tartrate|acetate|citrate|phosphate|maleate|fumarate|succinate|hydrobromide|besylate|tosylate)\b", re.I)
CATION = {"magnesium","calcium","sodium","potassium","zinc","iron","ferrous","ferric","lithium","aluminum","aluminium","ammonium","chloride","bicarbonate","carbonate","selenium","copper","manganese","strontium"}
def _clean(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()
def norm(s):
    s = re.sub(r"\(.*?\)", " ", (s or "").lower())
    base = _clean(DOSAGE.sub(" ", s)); salted = _clean(SALT.sub(" ", base))
    return base if (not salted or salted in CATION) else salted

con = sqlite3.connect(DB)
alias = {}; ambig = set(); src = Counter()
# curated synonyms (brand / research code / INN / USAN / FDA ...)
for ch, syn, st in con.execute("SELECT md.chembl_id, ms.synonyms, ms.syn_type FROM molecule_synonyms ms JOIN molecule_dictionary md ON ms.molregno=md.molregno WHERE ms.synonyms IS NOT NULL"):
    n = norm(syn)
    if len(n) < 3: continue
    if n in alias and alias[n] != ch: ambig.add(n)       # same string -> different molecule = ambiguous
    else: alias.setdefault(n, ch); src[st] += 1
# pref names (canonical INN-ish)
for ch, pn in con.execute("SELECT chembl_id, pref_name FROM molecule_dictionary WHERE pref_name IS NOT NULL"):
    n = norm(pn)
    if len(n) < 3: continue
    if n in alias and alias[n] != ch: ambig.add(n)
    else: alias.setdefault(n, ch); src["PREF_NAME"] += 1
# drop ambiguous strings (precision: don't ground a name that maps to multiple molecules)
for a in ambig: alias.pop(a, None)
json.dump(alias, open(A2 + "/chembl_aliases.json", "w"))
print(f"ChEMBL alias index: {len(alias)} unique name->chembl ({len(ambig)} ambiguous dropped)")
print("by source:", dict(src.most_common(8)))
# sanity: the brand/research-code names our LOO missed
for q in ["vidaza", "tylenol", "bi 10773", "premarin", "zantac", "herceptin"]:
    print(f"  {q!r} -> {alias.get(q)}")
