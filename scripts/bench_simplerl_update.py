"""Measure the gradient-side cost of a SimpleRL-style GRPO step on this box.

A GRPO step is three passes over the rollout batch beyond generation:
  1. actor  forward, no grad  -> old log-probs
  2. ref    forward, no grad  -> KL reference log-probs
  3. actor  forward + backward + optimizer step

This times a single micro-batch of each and reports tokens/s, so the per-step
cost is (tokens in the step) / (tokens per second) for each pass.
"""

import argparse
import json
import time
from pathlib import Path

import torch


def bench(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "Qwen2.5-1.5B"))
    ap.add_argument("--seqlen", type=int, default=1280)
    ap.add_argument("--micro-bs", type=int, nargs="+", default=[2, 8, 16])
    ap.add_argument("--attn", default=None, help="sdpa | flash_attention_2 | eager")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--logit-chunk", type=int, default=2,
                    help="rows of logits to log-softmax at once")
    ap.add_argument("--mem-fraction", type=float, default=0.5,
                    help="hard cap on the allocator, as a fraction of the "
                         "box's 121 GiB. Memory here is unified, so without a "
                         "cap an oversized batch does not raise "
                         "torch.OutOfMemoryError - it eats all system RAM and "
                         "freezes the machine.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    from transformers import AutoModelForCausalLM

    kw = {"dtype": torch.bfloat16}
    if args.attn:
        kw["attn_implementation"] = args.attn
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).cuda()
    attn_used = getattr(model.config, "_attn_implementation", "?")
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-7)

    V = model.config.vocab_size
    results = {
        "model": args.model,
        "mem_fraction": args.mem_fraction,
        "seqlen": args.seqlen,
        "attn_implementation": attn_used,
        "grad_checkpointing": not args.no_grad_ckpt,
        "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
        "passes": [],
    }

    for bs in args.micro_bs:
        ids = torch.randint(0, V, (bs, args.seqlen), device="cuda")
        tok = bs * args.seqlen
        row = {"micro_bs": bs, "tokens_per_micro_batch": tok}

        def fwd_nograd():
            # verl gathers the log-prob of the *taken* token; it never holds a
            # full fp32 [B, S, V] log-softmax. Chunking over the batch keeps the
            # peak at one row's logits and matches what the real trainer costs.
            with torch.no_grad():
                logits = model(ids).logits
                for i in range(0, logits.shape[0], args.logit_chunk):
                    chunk = logits[i : i + args.logit_chunk]
                    lp = torch.log_softmax(chunk.float(), dim=-1)
                    lp.gather(-1, ids[i : i + args.logit_chunk].unsqueeze(-1))

        def fwd_bwd():
            out = model(ids, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        try:
            torch.cuda.reset_peak_memory_stats()
            s = bench(fwd_nograd)
            row["logprob_fwd_s"] = round(s, 3)
            row["logprob_tok_per_s"] = round(tok / s)
            row["logprob_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        except torch.OutOfMemoryError:
            row["logprob_fwd_s"] = "OOM"
            torch.cuda.empty_cache()

        try:
            torch.cuda.reset_peak_memory_stats()
            s = bench(fwd_bwd)
            row["train_step_s"] = round(s, 3)
            row["train_tok_per_s"] = round(tok / s)
            row["train_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        except torch.OutOfMemoryError:
            row["train_step_s"] = "OOM"
            torch.cuda.empty_cache()

        print(json.dumps(row), flush=True)
        results["passes"].append(row)

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
