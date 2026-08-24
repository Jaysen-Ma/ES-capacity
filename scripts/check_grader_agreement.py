#!/usr/bin/env python3
"""Quantify how much the choice of grader changes the reward signal.

The GRPO arm is configured to use the ES arm's grader (`boxed_reward_fn`) rather than verl's
default (`hf_math_verify`). That is the right call for matching, but it is worth knowing the
size of the effect rather than asserting it: verl's `qwen_extract_answer` falls back through
"the answer is" -> "final answer is" -> THE LAST NUMBER IN THE STRING, so it can score 1.0 on
responses that never emit \\boxed{} at all.

This runs both graders over the same completions and reports the disagreement rate. Run it
before the real training run; put the number in the writeup.

Usage:
    python scripts/check_grader_agreement.py --n 200
    python scripts/check_grader_agreement.py --completions dump.jsonl   # score an existing dump
"""
import argparse
import json
import os
import sys

ES_AT_SCALE = os.environ.get("ES_AT_SCALE_PATH", "/workspace/repos/es-at-scale")
sys.path.insert(0, ES_AT_SCALE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--data", default="/workspace/data/lvl3to5_es_template/train.parquet")
    ap.add_argument("--completions", default=None,
                    help="jsonl with {response, ground_truth}; skips generation if given")
    ap.add_argument("--out", default="/workspace/data/grader_agreement.json")
    args = ap.parse_args()

    from es_at_scale.reward_function.math_grader import boxed_reward_fn

    try:
        from verl.utils.reward_score.hf_math_verify import compute_score as verl_score
        have_verl = True
    except Exception as e:
        print(f"note: verl grader unavailable ({e}); reporting ES grader stats only")
        have_verl = False

    samples = []
    if args.completions:
        with open(args.completions) as f:
            samples = [json.loads(l) for l in f][: args.n]
    else:
        import pandas as pd
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        df = pd.read_parquet(args.data).head(args.n)
        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = [tok.apply_chat_template(list(r["prompt"]), tokenize=False,
                                           add_generation_prompt=True)
                   for _, r in df.iterrows()]
        llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.6)
        # Match the training rollout config exactly — no stop tokens, T=1.0, 2048 cap.
        sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=2048)
        outs = llm.generate(prompts, sp)
        for (_, r), o in zip(df.iterrows(), outs):
            samples.append({"response": o.outputs[0].text,
                            "ground_truth": r["reward_model"]["ground_truth"]})

    agree = es_only = verl_only = both_zero = both_one = 0
    boxed = 0
    for s in samples:
        try:
            _m, es = boxed_reward_fn(s["response"], s["ground_truth"])
        except Exception:
            es = 0.0
        boxed += "\\boxed" in s["response"]
        if not have_verl:
            continue
        try:
            v = float(verl_score("simplelr_qwen", s["response"], s["ground_truth"]))
        except Exception:
            v = 0.0
        if es == v:
            agree += 1
            both_one += es == 1.0
            both_zero += es == 0.0
        elif es == 1.0:
            es_only += 1
        else:
            verl_only += 1

    n = len(samples)
    res = {
        "n": n,
        "frac_with_boxed": boxed / n if n else 0,
        "es_grader_pass_rate": None,
    }
    if have_verl:
        res.update({
            "agreement_rate": agree / n,
            "disagreement_rate": (es_only + verl_only) / n,
            "es_correct_verl_wrong": es_only,
            "verl_correct_es_wrong": verl_only,
            "both_correct": both_one,
            "both_wrong": both_zero,
        })
    print(json.dumps(res, indent=2))
    if have_verl and res["disagreement_rate"] > 0.05:
        print("\nNOTE: >5% disagreement. The grader choice materially changes the objective; "
              "state the number and the choice explicitly in the writeup.")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
