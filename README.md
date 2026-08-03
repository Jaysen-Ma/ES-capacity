# ES-capacity

**Does evolution-strategy post-training expand LLM reasoning capacity beyond the base model?**

This project generalises the pass@k capacity analysis of Yue et al. (NeurIPS 2025) from reinforcement learning with verifiable rewards (RLVR) to evolution-strategy (ES) post-training, using two recent ICML 2026 ES methods.

## Papers

| Role | Paper | Link |
|------|-------|------|
| Capacity analysis (pass@k) | Yue et al. | [arXiv:2504.13837](https://arxiv.org/abs/2504.13837) |
| Full-parameter ES | Qiu et al. | [arXiv:2509.24372](https://arxiv.org/abs/2509.24372) |
| Low-rank ES (EGGROLL) | Sarkar et al. | [arXiv:2511.16652](https://arxiv.org/abs/2511.16652) |

## Models

Primary checkpoints are named in `config.toml` (default: **Qwen2.5-7B** Base + Instruct). Paths are machine-local — do not hardcode them in scripts.

## Setup

```bash
cp config.sample.toml config.toml   # then edit paths.models_dir / paths.venv / models.*
source "$(python -m es_capacity.config venv)/bin/activate"   # or activate your venv first
pip install -e .
```

`config.toml` is gitignored. Commit only `config.sample.toml`.

## Fast inference: Docker vLLM (recommended)

Uses the image / port / max-len settings from `[vllm]` in `config.toml`.

```bash
# Wait until all .safetensors shards are present under the model dir, then:
bash scripts/serve_qwen_vllm_docker.sh base
docker logs -f es-capacity-vllm   # wait until Uvicorn is up
curl -s http://127.0.0.1:8000/v1/models
```

Evaluate (Yue `qwen-boxed` prompt + Yue `extract_answer`/`math_equal`):

```bash
python scripts/eval_aime_passk.py --model-key base
```

Then restart the container for Instruct:

```bash
bash scripts/serve_qwen_vllm_docker.sh instruct
python scripts/eval_aime_passk.py --model-key instruct
```

Eval defaults (`ks`, `n_samples`, `output_dir`, …) also come from `[eval]` in `config.toml`; CLI flags override them.

Smoke test: add `--ks 1 2 --n-samples 2 --max-new-tokens 2048`.

Fallbacks: `--backend vllm` (in-process) or `--backend hf`.

## Package map

```
config.sample.toml                  Sample local config (committed)
config.toml                         Local paths / knobs (gitignored)
scripts/serve_qwen_vllm_docker.sh   Docker vLLM launch
scripts/eval_aime_passk.py          AIME24 pass@k CLI
es_capacity/config.py               Config loader + `python -m es_capacity.config`
es_capacity/model.py                openai / vllm / hf backends
es_capacity/yue_math/               Vendored Yue parser + grader
es_capacity/reward.py               Yue extract_answer + math_equal
es_capacity/data.py                 AIME24 + qwen-boxed prompts
```

## Status

- **Runnable:** AIME24 pass@k via Docker vLLM OpenAI API + Yue grading.
- **Stubbed:** ES trainers / post-training pipeline.
