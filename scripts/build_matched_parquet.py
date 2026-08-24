#!/usr/bin/env python3
"""Rebuild the level3to5 training parquet with the ES arm's prompt template.

Why this exists: SimpleRL-Zoo's shipped train.parquet bakes the `qwen-boxed` template into
the prompt text (instruction in the USER turn) and contains a literal f-string bug —
`\\boxed{{}}` with doubled braces. The ES arm trained on `qwen_math_template` instead
(instruction in the SYSTEM turn, single braces). Matching the template is a precondition for
the ES-vs-GRPO comparison meaning anything.

verl v0.7.1 applies `tokenizer.apply_chat_template` to the `prompt` field, so we store
structured messages rather than a pre-rendered string. Verified byte-identical: applying
Qwen2.5-1.5B's chat template to these messages reproduces `qwen_math_template(question)`
exactly. The script asserts this at runtime rather than trusting it.
"""
import argparse
import os

SYSTEM_MSG = "Please reason step by step, and put your final answer within \\boxed{}."


def es_template(question: str) -> str:
    """Verbatim copy of es_at_scale.template_function.apply_template.qwen_math_template."""
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="https://huggingface.co/datasets/hkust-nlp/"
                    "SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5/train.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    df = pd.read_parquet(args.src)
    assert len(df) == 8523, f"expected 8523 rows (level3to5), got {len(df)}"

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Fail loudly if the chat template stops reproducing the ES string. A silent divergence
    # here would make the two arms train on different prompts with no visible symptom.
    probe = "What is $1+1$?"
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": probe}],
        tokenize=False, add_generation_prompt=True,
    )
    if rendered != es_template(probe):
        raise SystemExit(
            "chat template does not reproduce the ES template.\n"
            f"  ES  : {es_template(probe)!r}\n  chat: {rendered!r}\n"
            "Store pre-rendered text and disable verl's chat templating instead."
        )

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "data_source": "simplelr_qwen",          # routes to our custom reward fn
            "prompt": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": r["question"]},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(r["gt_answer"])},
            "extra_info": {"index": int(i), "level": int(r["level"]),
                           "question": r["question"], "split": "train"},
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)

    # Report the length distribution so max_prompt_length can be justified, not guessed.
    lens = [len(tok(es_template(q), add_special_tokens=False).input_ids) for q in df["question"]]
    over = {c: sum(1 for x in lens if x > c) for c in (1024, 2048)}
    print(f"wrote {len(out)} rows -> {args.out}")
    print(f"prompt tokens: mean={sum(lens)/len(lens):.1f} max={max(lens)}")
    print(f"  >1024: {over[1024]} ({100*over[1024]/len(lens):.3f}%)   "
          f">2048: {over[2048]} ({100*over[2048]/len(lens):.3f}%)")
    if over[2048]:
        print("WARNING: prompts exceed max_prompt_length=2048; they will be dropped, "
              "so the two arms would see different problem sets.")


if __name__ == "__main__":
    main()
