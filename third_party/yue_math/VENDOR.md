# Vendored: Yue et al. math evaluation (limit-of-RLVR)

| Field | Value |
|-------|-------|
| Upstream | https://github.com/LeapLabTHU/limit-of-RLVR (fork: Jaysen-Ma/limit-of-RLVR) |
| Commit | `79c348f4543330bb78b01a5332df09fea2700f70` |
| Commit date | 2026-07-27 |
| Vendored on | 2026-08-04 |
| Paper | arXiv:2504.13837 |
| Copied from | `math/examples/math_eval/` |
| **Licence** | **None stated upstream** — see below |

## Licence — unresolved

Upstream ships no `LICENSE` file and states no licence in its README, so by default all rights
are reserved and there is no explicit grant to redistribute this code, even unmodified.

This is almost certainly an oversight rather than intent: the repo exists so that others can
reproduce the paper's pass@k analysis, which is exactly what this project does, and it is a
very common omission in academic repos. We have asked upstream to add one — see the tracking
note in `.notes/DECISIONS.md`. Until they do:

- Use here is academic, non-commercial, and attributed.
- Keep the tree byte-identical to upstream except for the recorded patches, so it stays
  trivially removable if upstream ever objects.
- Do not assume this code is reusable under this repo's MIT licence. It is not ours to relicense.

To re-derive this tree, clone upstream, `git checkout 79c348f4`, copy the files listed
below out of `math/examples/math_eval/`, then apply `patches/*.patch` in numeric order
(`patch -p0` from this directory).

## Files copied

- `parser.py`, `grader.py`, `evaluate.py`, `utils.py`, `data_loader.py`
- `python_executor.py`, `examples.py`, `trajectory.py` (transitive imports)
- `latex2sympy/` (vendored LaTeX→SymPy package)

Minerva data lives at `data/minerva_math/test.jsonl` (SHA256 in that folder), not under this tree.

## Patches

| # | File | Description |
|---|------|-------------|
| 0001 | `evaluate.py` | Raise `ProcessPool(max_workers=...)` from 1 to 16 to avoid the grader hang observed 2026-08-03 (single worker + 3s timeout serialised all comparisons). |
| 0002 | `grader.py` | Fix a second grading deadlock observed 2026-08-04: `math_equal`'s equation-comparison branch (`pred.count("=")==1`) called `symbolic_equal(...)` directly, bypassing `call_with_timeout` even when `timeout=True` — unlike the "symbolic equal with sympy" fallback further down, which was already routed through it. A pathological equation-form completion could hang a grading worker forever with no bound. Also hardened `call_with_timeout` itself: the post-`terminate()` `process.join()` had no timeout, so a child that ignores `SIGTERM` (e.g. stuck deep in a C extension) could still wedge the parent forever; now it joins with a 5s timeout and escalates to `SIGKILL`. |

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
