# SimpleRL-Zoo training split — math_lvl3to5

8,523 problems. Downloaded from [`hkust-nlp/SimpleRL-Zoo-Data`](https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data)
(`simplelr_qwen_level3to5/train.parquet`) — the training set used for the GRPO arm
(`hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo`, arXiv:2503.18892).

Level distribution: {3: 2514, 4: 2648, 5: 3361}

The `prompt` field is the upstream training prompt verbatim. It matches the Yue et al.
`qwen-boxed` evaluation template except for a literal `\boxed{{}}` (double braces)
where the eval template has `\boxed{}`. The ES arm trains on this string so that both
post-training arms see the same prompt distribution; evaluation still uses the Yue
template for every arm.

Regenerate with:

```bash
python -m es_capacity.cli.fetch_train_data --split math_lvl3to5
```

SHA256: see `SHA256` in this directory.
