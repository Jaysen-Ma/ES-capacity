"""Fetch the GRPO baseline's training set into this repo's data layout.

The GRPO arm (`hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo`) was trained on MATH
level 3-5 via `hkust-nlp/SimpleRL-Zoo-Data`. Training the ES arm on the same
problems with the same prompt is what makes the comparison a comparison, so
this fetches that split rather than reconstructing it.

Note the upstream training prompt contains a literal `\\boxed{{}}` (an
unescaped format string in their pipeline) where the Yue et al. evaluation
template has `\\boxed{}`. We keep the upstream string verbatim: the ES arm
must train under the same prompt GRPO saw, not under the eval-time prompt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATASET_REPO = "hkust-nlp/SimpleRL-Zoo-Data"
SPLITS = {
    "math_lvl3to5": "simplelr_qwen_level3to5/train.parquet",
    "math_lvl1to4": "simplelr_qwen_level1to4/train.parquet",
    "gsm8k_math_lvl1": "simplelr_qwen_level1/train.parquet",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_split(split: str, out_root: Path, *, repo_root: Path) -> Path:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    if split not in SPLITS:
        raise KeyError(f"Unknown split {split!r}; known={list(SPLITS)}")

    src = hf_hub_download(repo_id=DATASET_REPO, filename=SPLITS[split], repo_type="dataset")
    df = pd.read_parquet(src)

    out_dir = out_root / f"simplerl_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "train.jsonl"

    with jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(df.itertuples(index=False)):
            record = {
                "idx": idx,
                "problem": row.question,
                "answer": str(row.gt_answer),
                "level": int(row.level),
                "subject": row.subject,
                # Verbatim training prompt as GRPO saw it, including the `{{}}` artifact.
                "prompt": row.prompt[0]["content"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    digest = _sha256(jsonl)
    rel = jsonl.relative_to(repo_root) if jsonl.is_relative_to(repo_root) else jsonl
    (out_dir / "SHA256").write_text(f"{digest}  {rel}\n")

    levels = df["level"].value_counts().sort_index().to_dict()
    (out_dir / "README.md").write_text(
        f"""# SimpleRL-Zoo training split — {split}

{len(df):,} problems. Downloaded from [`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO})
(`{SPLITS[split]}`) — the training set used for the GRPO arm
(`hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo`, arXiv:2503.18892).

Level distribution: {levels}

The `prompt` field is the upstream training prompt verbatim. It matches the Yue et al.
`qwen-boxed` evaluation template except for a literal `\\boxed{{{{}}}}` (double braces)
where the eval template has `\\boxed{{}}`. The ES arm trains on this string so that both
post-training arms see the same prompt distribution; evaluation still uses the Yue
template for every arm.

Regenerate with:

```bash
python -m es_capacity.cli.fetch_train_data --split {split}
```

SHA256: see `SHA256` in this directory.
"""
    )
    print(f"wrote {jsonl} ({len(df):,} problems, {jsonl.stat().st_size/1e6:.1f} MB)")
    print(f"sha256={digest}")
    return jsonl


def main(argv: list[str] | None = None) -> None:
    import argparse

    from es_capacity.config import load_config

    p = argparse.ArgumentParser(description="Download the GRPO baseline's training data")
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    p.add_argument("--split", default="math_lvl3to5", choices=sorted(SPLITS))
    p.add_argument("--data-dir", default=None, help="Defaults to config paths.data_dir")
    args = p.parse_args(argv)

    cfg = load_config(machine=args.machine)
    out_root = Path(args.data_dir) if args.data_dir else cfg.data_dir
    fetch_split(args.split, out_root, repo_root=cfg.repo_root)


if __name__ == "__main__":
    main()
