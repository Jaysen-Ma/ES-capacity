# Vendored: EGGROLL + vLLM (eggroll-vllm)

| Field | Value |
|-------|-------|
| Upstream | https://github.com/ESHyperscale/eggroll-vllm (fork: Jaysen-Ma/eggroll-vllm) |
| Local clone | `/home/zotar/jupyterlab/eggroll-vllm` |
| Commit | `bcc215e8784f5f44d24985145c0a71e74283cf1f` |
| Paper | arXiv:2511.16652 |

## Files copied

- `es_lora_multinode.py`, `es_lora_multinode_moe.py`
- `tasks.py`, `merge_checkpoint.py`, `countdown.json`
- `requirements.txt`, `README.md`, launch scripts, `Dockerfile`

## Known upstream issues (document; patch when wiring)

1. `tasks.py` imports missing `egg_img` module — **patched** `0001-tasks-stub-egg-img.patch` (stub empty strings).
2. `merge_checkpoint.py` references task classes absent from `tasks.py`.
3. Pins vLLM 0.17.0; we run 0.26.0 — probe Multi-LoRA + `worker_extension_cls` before training.
4. Train reward should call `es_capacity.grade` (Yue grader), not `gem-llm`.

## Usage in this repo

Orchestration lives in `es_capacity/posttrain/`. Prefer importing algorithm helpers from here rather than rewriting low-rank noise math.
