"""Track A Step 3: SFT a tiny student as the band reranker (peft LoRA + transformers.Trainer; no trl/bnb).
Task = closed-set pick: given a condition string + 8 SapBERT candidate concepts (with synonyms) + '0 = none',
output the gold's number, or 0 to abstain (gold not in top-8). Reward/target is the verifiable MONDO id.
Trains on band_train.jsonl; evals BASE vs TRAINED on band_eval.jsonl (full pick-or-abstain accuracy) +
the frozen band_cases.json (continuity vs SapBERT 0.750 / Qwen-7B 0.810 / Claude 0.900 / ceiling 0.970).
Usage: sft_train.py <model> [epochs]"""
import os, re, sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
A2 = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
EPOCHS = float(sys.argv[2]) if len(sys.argv) > 2 else 2
dev = "cuda" if torch.cuda.is_available() else "cpu"

def fmt(rec):
    lines = []
    for i, c in enumerate(rec["candidates"]):
        s = "; ".join(c.get("syns", [])[:4]); lines.append(f"{i+1}. {c['name']}" + (f"  [aka: {s}]" if s else ""))
    return (f'Match the clinical-trial condition string to ONE disease concept.\n'
            f'Condition: "{rec["query"]}"\nCandidates:\n' + "\n".join(lines) +
            f'\n0. none of these\nReply with only the number (0-{len(rec["candidates"])}):')
def gold_pos(rec):
    for i, c in enumerate(rec["candidates"]):
        if c["mondo_id"] == rec["gold"]: return i + 1
    return 0

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def build(rec):  # full chat string with the target appended (for SFT) + the prompt-only length (to mask)
    msgs = [{"role": "user", "content": fmt(rec)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = p + str(gold_pos(rec)) + tok.eos_token
    return p, full

class DS(torch.utils.data.Dataset):
    def __init__(self, recs): self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        p, full = build(self.recs[i])
        ids = tok(full, truncation=True, max_length=512, return_tensors="pt").input_ids[0]
        plen = len(tok(p, truncation=True, max_length=512).input_ids)
        lab = ids.clone(); lab[:plen] = -100                      # train only on the answer token(s)
        return {"input_ids": ids, "labels": lab}
def collate(b):
    m = max(len(x["input_ids"]) for x in b)
    iid = torch.full((len(b), m), tok.pad_token_id); lab = torch.full((len(b), m), -100); att = torch.zeros((len(b), m), dtype=torch.long)
    for k, x in enumerate(b):
        n = len(x["input_ids"]); iid[k, :n] = x["input_ids"]; lab[k, :n] = x["labels"]; att[k, :n] = 1
    return {"input_ids": iid, "labels": lab, "attention_mask": att}

train = [json.loads(l) for l in open(A2 + "/band_train.jsonl")]
ev = [json.loads(l) for l in open(A2 + "/band_eval.jsonl")][:1000]
bc = json.load(open(A2 + "/band_cases.json"))
print(f"{MODEL} | train {len(train)} | eval {len(ev)} + band_cases {len(bc)}", flush=True)

@torch.no_grad()
def evaluate(model, recs, abstain_aware=True):
    model.eval(); ok = 0
    for r in recs:
        p, _ = build(r); inp = tok(p, return_tensors="pt", truncation=True, max_length=512).to(dev)
        out = model.generate(**inp, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id)
        a = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True); m = re.search(r"\d+", a); pick = int(m.group()) if m else -1
        gp = gold_pos(r)
        ok += int(pick == gp) if abstain_aware else int(pick == gp and gp > 0)
    return ok / len(recs)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
base_ev = evaluate(model, ev); base_bc = evaluate(model, bc, abstain_aware=False)
print(f"BASE  : band_eval(full acc) {base_ev:.3f} | band_cases(pick) {base_bc:.3f}", flush=True)

model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM"))
args = TrainingArguments(output_dir=f"/tmp/sft_{MODEL.split('/')[-1]}", num_train_epochs=EPOCHS, per_device_train_batch_size=16,
    gradient_accumulation_steps=2, learning_rate=1e-4, bf16=True, logging_steps=50, save_strategy="no", report_to=[], warmup_ratio=0.03)
Trainer(model=model, args=args, train_dataset=DS(train), data_collator=collate).train()

tr_ev = evaluate(model, ev); tr_bc = evaluate(model, bc, abstain_aware=False)
adir = A2 + f"/adapter_{MODEL.split('/')[-1]}"; model.save_pretrained(adir); print(f"saved adapter -> {adir}", flush=True)
print(f"\n=== SFT RESULT ({MODEL}) ===")
print(f"  band_eval (full pick-or-abstain acc) : base {base_ev:.3f} -> SFT {tr_ev:.3f}")
print(f"  band_cases (pick, vs SapBERT 0.750 / Qwen-7B 0.810 / Claude 0.900 / ceiling 0.970): base {base_bc:.3f} -> SFT {tr_bc:.3f}")
