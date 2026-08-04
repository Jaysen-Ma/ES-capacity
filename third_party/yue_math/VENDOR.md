# Vendored: Yue et al. math evaluation (limit-of-RLVR)

| Field | Value |
|-------|-------|
| Upstream | https://github.com/Yueyang130/limit-of-RLVR (fork: Jaysen-Ma/limit-of-RLVR) |
| Local clone | `/home/zotar/jupyterlab/limit-of-RLVR` |
| Commit | `79c348f4543330bb78b01a5332df09fea2700f70` |
| Paper | arXiv:2504.13837 |
| Copied from | `math/examples/math_eval/` |

## Files copied

- `parser.py`, `grader.py`, `evaluate.py`, `utils.py`, `data_loader.py`
- `python_executor.py`, `examples.py`, `trajectory.py` (transitive imports)
- `latex2sympy/` (vendored LaTeX→SymPy package)

Minerva data lives at `data/minerva_math/test.jsonl` (SHA256 in that folder), not under this tree.

## Patches

| # | File | Description |
|---|------|-------------|
| 0001 | `evaluate.py` | Raise `ProcessPool(max_workers=...)` from 1 to 16 to avoid the grader hang observed 2026-08-03 (single worker + 3s timeout serialised all comparisons). |

Our primary grading path is `es_capacity.grade`, which calls `math_equal` / `extract_answer` with its own pool; this patch keeps the upstream `evaluate()` usable if invoked directly.

## Sympy version note

Upstream `eval_math_nodes.sh` pins `sympy==1.12`. Torch 2.11 / vLLM 0.26 on this box require `sympy>=1.13` (import `_args_sortkey`). We run **sympy 1.14** and smoke-tested `extract_answer` / `math_equal` successfully. Prefer not forcing 1.12 into the inference environment.

## Install latex2sympy

```bash
cd third_party/yue_math/latex2sympy
pip install -e . --use-pep517 --no-deps
# antlr4-python3-runtime==4.11.1 recommended; sympy: use env-compatible version (see above)
```

## Do not

Refactor these files in place. Add a new numbered patch instead.
