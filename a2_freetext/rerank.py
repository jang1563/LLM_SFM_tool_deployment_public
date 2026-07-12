"""BandReranker: deployable precision recovery for the disease resolver's 0.85-0.95 ambiguous band.
Validated by reranker_proto2.py: closed-set LLM rerank of SapBERT's top-8 candidates, EACH shown with its
de-leaked synonym set. Corrected band precision: SapBERT 0.750, local SFT-1.5B 0.875, Claude-haiku 0.900,
gold-in-top8 ceiling 0.970.
It only RE-RANKS retrieved candidate ids (or abstains) — never generates an id — so it stays consistent with
the 'retrieve, don't hallucinate' thesis. Model is pluggable: API reference or local open-model LoRA."""
import re, os, json, numpy as np
from pathlib import Path
from resolve_disease import DiseaseResolver, _norm
A2 = str(Path(os.environ.get("A2_FREETEXT_DIR", Path(__file__).resolve().parent)).expanduser().resolve())
GEN = {"disease","diseases","disorder","disorders","syndrome","syndromes","cancer","neoplasm","tumor","tumour","infection","carcinoma","malignancy","abnormality","condition","deficiency","injury","failure","pain","lesion","complication"}


def _load_id2forms():
    id2forms = {}; cur = None
    for line in open(A2 + "/mondo.obo"):
        line = line.rstrip("\n")
        if line == "[Term]": cur = None
        elif line.startswith("id: "): cur = line[4:]
        elif cur and cur.startswith("MONDO:"):
            if line.startswith("is_obsolete: true"): id2forms.pop(cur, None); cur = None
            elif line.startswith("name: "):
                n = _norm(line[6:])
                if len(n) >= 4 and n not in GEN: id2forms.setdefault(cur, set()).add(n)
            elif line.startswith("synonym: "):
                m = re.match(r'synonym: "(.*?)" (EXACT|NARROW)', line)
                if m:
                    n = _norm(m.group(1))
                    if len(n) >= 4 and n not in GEN: id2forms.setdefault(cur, set()).add(n)
    return id2forms


class BandReranker:
    """backend='anthropic' (Claude reference, 0.900 de-leaked) or 'local' (SFT tiny open model + LoRA adapter,
    e.g. Qwen2.5-1.5B at 0.875 de-leaked band_cases). The 'local' prompt MUST match the sft_train.py format."""
    def __init__(self, backend="anthropic", model="claude-haiku-4-5", base="Qwen/Qwen2.5-1.5B-Instruct", adapter_dir=None):
        self.backend, self.model = backend, model
        self.id2forms = _load_id2forms()
        if backend == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif backend == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(base)
            m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
            if adapter_dir:
                from peft import PeftModel
                m = PeftModel.from_pretrained(m, adapter_dir)
            self.dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.lm = m.to(self.dev).eval(); self.torch = torch

    def _prompt(self, query, cand_ids, names):  # MUST match sft_train.py:fmt (both de-leaked)
        lines = []
        for i, c in enumerate(cand_ids):
            # DE-LEAK: exclude the query itself from displayed syns (else the answer is shown verbatim)
            syns = sorted(s for s in self.id2forms.get(c, []) if s != _norm(names.get(c, "")) and s != _norm(query))[:4]
            lines.append(f"{i+1}. {names.get(c, c)}" + (f"  [aka: {'; '.join(syns)}]" if syns else ""))
        return (f'Match the clinical-trial condition string to ONE disease concept.\n'
                f'Condition: "{query}"\nCandidates:\n' + "\n".join(lines) +
                f'\n0. none of these\nReply with only the number (0-{len(cand_ids)}):')

    def rerank(self, query, cand_ids, names):
        """cand_ids: ordered list of MONDO ids (SapBERT top-k). Returns (mondo_id, name) or None (abstain)."""
        prompt = self._prompt(query, cand_ids, names)
        if self.backend == "anthropic":
            try:
                r = self.client.messages.create(model=self.model, max_tokens=8, messages=[{"role": "user", "content": prompt}])
                txt = r.content[0].text
            except Exception:
                return None
        else:  # local zero-API
            t = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inp = self.tok(t, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                out = self.lm.generate(**inp, max_new_tokens=4, do_sample=False, pad_token_id=self.tok.eos_token_id)
            txt = self.tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"\d+", txt); pick = int(m.group()) if m else 0
        if 1 <= pick <= len(cand_ids):
            c = cand_ids[pick - 1]; return (c, names.get(c, c))
        return None


class RerankedDiseaseResolver:
    """DiseaseResolver + the band reranker: >=0.95 trusted as-is (92-95% precise), <0.85 abstain, and the
    0.85-0.95 band (50-66% precise, ~29% of resolutions) routed through the reranker."""
    def __init__(self, base=None, reranker=None, lo=0.85, hi=0.95, k=8):
        self.base = base or DiseaseResolver()
        self.reranker = reranker or BandReranker()
        self.lo, self.hi, self.k = lo, hi, k
        self.hier, self.idname = self.base.hier, self.base.idname   # delegate for A2Resolver grounding
        self.mondo2conds = self.base.mondo2conds

    def ground(self, mondo_id): return self.base.ground(mondo_id)

    def _topk(self, text):
        q = self.base._embed(_norm(text)); sims = (q @ self.base.Lmat.T)[0]; order = np.argsort(-sims)
        cands = []; seen = set()
        for j in order:
            mid = self.base.lab2id[self.base.labels[j]]
            if self.base._obsolete(mid) or mid in seen: continue
            seen.add(mid); cands.append(mid)
            if len(cands) >= self.k: break
        return cands, float(sims[order[0]])

    def resolve(self, text):
        r = self.base.resolve(text)
        if "abstain" in r or r.get("source") == "dict" or r.get("conf", 0) >= self.hi:
            return r                                        # exact / confident / abstain -> unchanged
        cands, top_sim = self._topk(text)
        if not (self.lo <= top_sim < self.hi):
            return r
        picked = self.reranker.rerank(text, cands, self.base.idname)
        if picked is None:
            return {"abstain": f"reranker rejected all {len(cands)} candidates (band {top_sim:.2f})"}
        mid, name = picked
        return {"mondo_id": mid, "name": name, "conf": round(top_sim, 3), "source": "reranked", "base_top1": r.get("name")}


if __name__ == "__main__":
    R = RerankedDiseaseResolver(); print("loaded.\n")
    # find a few band cases where SapBERT top-1 is WRONG, show base vs reranked
    import random
    rng = random.Random(1)
    base = R.base
    # rebuild the held-out band test to pull demo cases
    id2forms = _load_id2forms()
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
    shown = 0
    for k, (h, gold) in enumerate(test):
        sims = Tq[k] @ I.T; j = int(sims.argmax()); s = float(sims[j])
        if 0.85 <= s < 0.95 and tnmid[j] != gold:        # SapBERT top-1 wrong, in band
            r = R.resolve(h)
            print(f'query: "{h}"\n  gold        : {base.idname.get(gold)}\n  SapBERT top1: {base.idname.get(tnmid[j])}'
                  f'\n  reranked    : {r.get("name") if "abstain" not in r else "ABSTAIN"}  {"✓" if r.get("mondo_id")==gold else ("(abstain)" if "abstain" in r else "✗")}\n')
            shown += 1
            if shown >= 6: break
