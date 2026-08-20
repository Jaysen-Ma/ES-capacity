"""Turn the two measured benchmarks into a wall-clock estimate for a GRPO run.

A GRPO step on one GPU is four serial phases over the same batch of
`prompts x n` sequences:

    generate -> old log-probs (actor, no grad) -> ref log-probs -> update

plus per-step overhead verl pays regardless (weight sync into the vLLM engine,
rule-based grading, dataloading). Every rate below comes from
bench_simplerl_rollout.py / bench_simplerl_update.py on this box; nothing here
is a FLOP-count guess.
"""

import argparse
import json
from pathlib import Path


def fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-json", required=True)
    ap.add_argument("--update-json", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--prompts", type=int, default=256)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--overhead-s", type=float, default=90.0,
                    help="per-step weight sync + grading + dataloader")
    ap.add_argument("--micro-bs", type=int, default=None,
                    help="which micro-batch row of the update bench to use "
                         "(default: the fastest that did not OOM)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    roll = json.loads(Path(args.rollout_json).read_text())
    upd = json.loads(Path(args.update_json).read_text())

    rows = [r for r in upd["passes"] if isinstance(r.get("train_tok_per_s"), int)]
    if args.micro_bs is not None:
        rows = [r for r in rows if r["micro_bs"] == args.micro_bs]
    row = max(rows, key=lambda r: r["train_tok_per_s"])

    seqs = args.prompts * args.n
    resp_len = roll["resp_len_mean"]
    prompt_len = roll["prompt_len_mean"]
    seq_tokens = prompt_len + resp_len
    step_tokens = seqs * seq_tokens

    # Generation is measured at its own step shape; scale by sequence count so a
    # pilot run of a fraction of a step still gives the right per-step number.
    gen_s = seqs * resp_len / roll["output_tok_per_s"]
    old_lp_s = step_tokens / row["logprob_tok_per_s"]
    ref_lp_s = old_lp_s
    update_s = step_tokens / row["train_tok_per_s"]
    step_s = gen_s + old_lp_s + ref_lp_s + update_s + args.overhead_s

    est = {
        "config": {
            "steps": args.steps,
            "prompts_per_step": args.prompts,
            "rollouts_per_prompt": args.n,
            "sequences_per_step": seqs,
            "total_generations": seqs * args.steps,
            "max_tokens": roll["max_tokens"],
            "measured_mean_response_len": resp_len,
            "measured_mean_prompt_len": prompt_len,
            "tokens_per_step": int(step_tokens),
            "update_micro_bs": row["micro_bs"],
        },
        "rates": {
            "generate_tok_per_s": roll["output_tok_per_s"],
            "logprob_tok_per_s": row["logprob_tok_per_s"],
            "train_tok_per_s": row["train_tok_per_s"],
        },
        "per_step_s": {
            "generate": round(gen_s),
            "old_log_prob": round(old_lp_s),
            "ref_log_prob": round(ref_lp_s),
            "update": round(update_s),
            "overhead": args.overhead_s,
            "total": round(step_s),
        },
        "per_step": fmt_hms(step_s),
        "total": fmt_hms(step_s * args.steps),
        "total_hours": round(step_s * args.steps / 3600, 1),
        "total_days": round(step_s * args.steps / 86400, 2),
    }
    print(json.dumps(est, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(est, indent=2) + "\n")


if __name__ == "__main__":
    main()
