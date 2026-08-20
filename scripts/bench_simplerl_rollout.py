"""Measure vLLM rollout throughput for a SimpleRL-style GRPO step on this box.

One GRPO step generates `prompts x n` samples. This script runs exactly that
shape and reports the wall time, so a 50-step run can be extrapolated from a
measurement rather than from a FLOP count.

Prompts come from the same math_lvl3to5_8k set the ES runs trained on, already
rendered with the Qwen chat template in the `input` column.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

DEFAULT_DATA = "/home/zotar/jupyterlab/es-at-scale/datasets/train/math_lvl3to5_8k"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "Qwen2.5-1.5B"))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--n", type=int, default=32, help="rollouts per prompt")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    # NOTE: on this unified-memory box gpu_memory_utilization is a fraction of
    # all 121 GiB of system RAM, not of a separate VRAM pool. 0.85 reserves
    # ~103 GiB and leaves nothing for anything else running.
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_from_disk
    from vllm import LLM, SamplingParams

    ds = load_from_disk(args.data)["train"]
    ds = ds.shuffle(seed=args.seed).select(range(args.prompts))
    prompts = list(ds["input"])

    t_init = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_len + args.max_tokens,
        seed=args.seed,
        enforce_eager=False,
    )
    init_s = time.time() - t_init

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    lens, capped = [], 0
    prompt_lens = []
    for o in outs:
        prompt_lens.append(len(o.prompt_token_ids))
        for c in o.outputs:
            lens.append(len(c.token_ids))
            capped += c.finish_reason == "length"

    n_seq = len(lens)
    out_tok = sum(lens)
    res = {
        "model": args.model,
        "prompts": args.prompts,
        "n": args.n,
        "sequences": n_seq,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "engine_init_s": round(init_s, 1),
        "generate_s": round(gen_s, 1),
        "output_tokens": out_tok,
        "output_tok_per_s": round(out_tok / gen_s, 1),
        "seq_per_s": round(n_seq / gen_s, 2),
        "resp_len_mean": round(statistics.mean(lens), 1),
        "resp_len_median": statistics.median(lens),
        "resp_len_p90": sorted(lens)[int(0.9 * n_seq)],
        "resp_len_max": max(lens),
        "frac_hit_cap": round(capped / n_seq, 4),
        "prompt_len_mean": round(statistics.mean(prompt_lens), 1),
        "total_tokens_incl_prompt": out_tok + sum(prompt_lens) * args.n,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
