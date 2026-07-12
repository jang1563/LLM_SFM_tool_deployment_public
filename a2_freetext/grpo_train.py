"""Track B: from-base GRPO on the band reranker (the two-walls diagnostic probe). Single-token pick
(0-8) = contextual bandit, so manual GRPO (no trl): per prompt sample G picks from the answer-token dist,
reward = +1 iff pick==gold_pos (verifiable; gold_pos=0 means abstain/gold-not-in-top8), advantage =
group-normalized, loss = -(adv*logprob) + beta*KL(policy||frozen-base). LOG THE STARVATION SIGNATURE:
frac_reward_zero_std (all G picks same reward -> no signal, the original task's failure) + grad_norm.
Corrected de-leaked interpretation: SFT installs the hard in-band disambiguation; from-base GRPO improved the
full pick-or-abstain task mainly by learning the easier abstain gate, not by lifting band_cases.
Usage: grpo_train.py <model> [steps]"""
import os, re, sys, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
A2 = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
dev = "cuda" if torch.cuda.is_available() else "cpu"
G, BS, BETA, LR = 8, 16, 0.02, 1e-5     # samples/prompt, prompts/step, KL coef, lr

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
tok.padding_side = "left"
ANS = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(9)]   # token ids for "0".."8"
train = [json.loads(l) for l in open(A2 + "/band_train.jsonl")]
ev = [json.loads(l) for l in open(A2 + "/band_eval.jsonl")][:600]
bc = json.load(open(A2 + "/band_cases.json"))
print(f"{MODEL} | GRPO from base | train {len(train)} | ANS tokens {ANS}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM"))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

def ans_logits(recs):  # [B, 9] logits over answer tokens at the generation position
    prompts = [tok.apply_chat_template([{"role":"user","content":fmt(r)}], tokenize=False, add_generation_prompt=True) for r in recs]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
    out = model(**enc).logits[:, -1, :]            # left-padded -> last col is the next-token position
    return out[:, ANS]

@torch.no_grad()
def evaluate(recs, abstain_aware=True):
    model.eval(); ok = 0
    for s in range(0, len(recs), 32):
        b = recs[s:s+32]; pick = ans_logits(b).argmax(-1).tolist()
        for r, p in zip(b, pick):
            gp = gold_pos(r); ok += int(p == gp) if abstain_aware else int(p == gp and gp > 0)
    model.train(); return ok / len(recs)

base_bc = evaluate(bc, False); base_ev = evaluate(ev)
print(f"BASE: band_cases(pick) {base_bc:.3f} | band_eval(full) {base_ev:.3f}", flush=True)

import random; rng = random.Random(0); model.train()
for step in range(STEPS):
    recs = [train[rng.randrange(len(train))] for _ in range(BS)]
    gp = torch.tensor([gold_pos(r) for r in recs], device=dev)
    logits = ans_logits(recs); logp_all = torch.log_softmax(logits.float(), -1)        # [B,9]
    with torch.no_grad():
        with model.disable_adapter(): ref_logp = torch.log_softmax(ans_logits(recs).float(), -1)
        picks = torch.multinomial(logp_all.exp(), G, replacement=True)                  # [B,G]
        rew = (picks == gp[:, None]).float()                                            # verifiable reward
        adv = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-6)     # group-normalized
        frac_zero_std = (rew.std(1) < 1e-6).float().mean().item()                       # STARVATION signature
    lp = logp_all.gather(1, picks)                                                      # [B,G] logprob of picks
    kl = (logp_all.exp() * (logp_all - ref_logp)).sum(-1).mean()
    loss = -(adv * lp).mean() + BETA * kl
    opt.zero_grad(); loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0).item()
    opt.step()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"  step {step} | loss {loss.item():.3f} | reward {rew.mean().item():.3f} | frac_reward_zero_std {frac_zero_std:.3f} | grad_norm {gnorm:.3f} | KL {kl.item():.3f}", flush=True)

tr_bc = evaluate(bc, False); tr_ev = evaluate(ev)
print(f"\n=== GRPO RESULT ({MODEL}) ===")
print(f"  band_cases (pick, vs SFT 0.875 / SapBERT 0.750 / ceiling 0.970): base {base_bc:.3f} -> GRPO {tr_bc:.3f}")
print(f"  band_eval  (full pick-or-abstain, vs SFT 0.773)                 : base {base_ev:.3f} -> GRPO {tr_ev:.3f}")
delta_bc = tr_bc - base_bc
delta_ev = tr_ev - base_ev
verdict = "IN_BAND_LIFT" if delta_bc > 0.03 else "NO_IN_BAND_LIFT"
print(f"  -> diagnostic: {verdict}; band_eval delta {delta_ev:+.3f} should be interpreted separately as abstain-gate learning")
