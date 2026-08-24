#!/usr/bin/env python3
"""Measure KL(model || base) offline, per arm.

Why offline: the GRPO arm runs with both of verl's KL paths zeroed (use_kl_loss=False,
use_kl_in_reward=False), because a KL anchor to the base model mechanically preserves pass@k
and would make "GRPO preserved the ceiling" tautological. But verl only constructs the
reference-policy worker when KL training is enabled, so it never logs KL either. We want KL
as a MEASUREMENT, not as a training signal — hence this script.

Why it matters: the ES-at-scale paper's Table 2 reports mean KL from base at exactly this
project's ES hyperparameters (sigma=0.001, alpha=0.0005) on Qwen2.5-7B:

    ES (sigma=0.001)   0.274
    GRPO (beta=0.0)    0.861
    GRPO (beta=0.0167) 1.591

ES sits 3-6x closer to base than GRPO at any beta they tested. So "ES preserves the pass@k
ceiling" has an unexcluded null hypothesis: ES barely moved the weights. Reporting achieved
KL per arm — and plotting reward against KL rather than single points — is what makes the
headline claim falsifiable.

Usage:
    python scripts/measure_kl.py --model <ckpt-or-hf-id> --base Qwen/Qwen2.5-1.5B --n 512
"""
import argparse
import json

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the post-trained arm (HF format)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--data", default="/workspace/data/lvl3to5_es_template/train.parquet",
                    help="prompts to evaluate over; use a HELD-OUT split for a clean number")
    ap.add_argument("--n", type=int, default=512, help="number of prompts")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sample at the arm's TRAINING temperature; KL is distribution-dependent")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    df = pd.read_parquet(args.data).head(args.n)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda:1")
    model.eval(); base.eval()

    total_kl, total_tokens = 0.0, 0
    per_prompt = []

    for _, row in df.iterrows():
        msgs = list(row["prompt"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")

        # Sample a continuation FROM THE TRAINED MODEL, then score both models on it.
        # KL is measured on-policy w.r.t. the arm being evaluated.
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new,
                                 do_sample=args.temperature > 0,
                                 temperature=args.temperature or 1.0, top_p=1.0,
                                 pad_token_id=tok.eos_token_id)
            resp = gen[:, ids.shape[1]:]
            if resp.shape[1] == 0:
                continue
            full = gen
            lp_m = torch.log_softmax(model(full).logits[:, ids.shape[1]-1:-1].float(), -1)
            lp_b = torch.log_softmax(base(full.to("cuda:1")).logits[:, ids.shape[1]-1:-1].float(), -1)

        # Forward KL, summed over vocab, averaged over response tokens.
        kl = (lp_m.exp() * (lp_m - lp_b.to("cuda:0"))).sum(-1)
        total_kl += kl.sum().item()
        total_tokens += kl.numel()
        per_prompt.append(kl.mean().item())

    mean_kl = total_kl / max(total_tokens, 1)
    result = {
        "model": args.model, "base": args.base, "temperature": args.temperature,
        "n_prompts": len(per_prompt), "n_tokens": total_tokens,
        "mean_kl_per_token": mean_kl,
        "median_kl_per_prompt": sorted(per_prompt)[len(per_prompt) // 2] if per_prompt else None,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
