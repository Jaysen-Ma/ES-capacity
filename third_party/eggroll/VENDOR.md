# Vendored: EGGROLL + vLLM (eggroll-vllm)

| Field | Value |
|-------|-------|
| Upstream | https://github.com/ESHyperscale/eggroll-vllm (fork: Jaysen-Ma/eggroll-vllm) |
| Commit | `bcc215e8784f5f44d24985145c0a71e74283cf1f` |
| Commit date | 2026-04-14 |
| Vendored on | 2026-08-04 |
| Paper | arXiv:2511.16652 |
| **Licence** | **GPL-3.0** — see `LICENSE` in this directory |

## Licence

GPL-3.0, copied verbatim to `LICENSE` here. This is **not** the repo's MIT licence and it is
copyleft: anything distributed that is a derivative of these files must itself be GPL-3.0.

Practical consequences for this repo:

- The `patches/` convention doubles as GPL-3.0 §5(a) compliance — modified files carry a
  prominent, dated record of what changed. Keep using it.
- `es_capacity/` must not become a derivative of this tree if we want to keep our code MIT.
  Reimplementing an algorithm described in the paper is fine (algorithms are not
  copyrightable); porting or adapting these source files is not. `es_capacity/posttrain/`
  therefore works from arXiv:2511.16652, not from `es_lora_multinode.py`, and cites the paper.

To re-derive this tree, clone upstream, `git checkout bcc215e8`, copy the files listed
below, then apply `patches/*.patch` in numeric order (`patch -p0` from this directory).

## Files copied

- `es_lora_multinode.py`, `es_lora_multinode_moe.py`
- `tasks.py`, `merge_checkpoint.py`, `countdown.json`
- `requirements.txt`, `README.md`, launch scripts, `Dockerfile`

## Patches

| # | File | Description |
|---|------|-------------|
| 0001 | `tasks.py` | Upstream imports `from egg_img import EGG_IMG, CHICK_IMG`, but `egg_img.py` is not in the repo — the module is purely cosmetic ASCII art for progress logging, so importing `tasks.py` fails outright. Stub both names to empty strings. |

## Known upstream issues (document; patch when wiring)

1. `tasks.py` imports missing `egg_img` module — **patched**, see 0001 above.
2. `merge_checkpoint.py` references task classes absent from `tasks.py`.
3. Pins vLLM 0.17.0; we run 0.26.0 — probe Multi-LoRA + `worker_extension_cls` before training.
4. Train reward should call `es_capacity.grade` (Yue grader), not `gem-llm`.

## Do not

Refactor these files in place. Add a new numbered patch instead.

## Usage in this repo

Orchestration lives in `es_capacity/posttrain/`. Prefer importing algorithm helpers from here rather than rewriting low-rank noise math.
