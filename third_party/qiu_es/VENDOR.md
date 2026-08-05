# Vendored: Qiu et al. ES-at-Scale (full-rank ES)

| Field | Value |
|-------|-------|
| Upstream | https://github.com/VsonicV/es-at-scale (fork: Jaysen-Ma/es-at-scale) |
| Commit | `574a9d134da1ffce2a8bb812019899e5c96b588a` |
| Commit date | 2026-06-26 |
| Vendored on | 2026-08-04 |
| Paper | arXiv:2509.24372 |

To re-derive this tree, clone upstream, `git checkout 574a9d13`, and copy the files
listed below. There are no patches — this tree is byte-identical to upstream.

## Files copied

- `es_at_scale/` package (`train.py`, `trainer/es_trainer.py`, `utils/worker_extension.py`, reward + template modules)
- `setup.py`, `README.md`, `LICENSE.txt`

Training datasets remain in the sibling clone (`../es-at-scale/datasets/`) to avoid duplicating large files.

## Patches

None. `patches/` is empty and this tree is byte-identical to upstream at the commit above.

## Role in v1

Interface-only / v2 drop-in. EGGROLL is the v1 ES arm. `es_capacity.posttrain.qiu` wraps the full-rank Perturber (`torch.randn` in-place via `worker_extension_cls`) without a full training loop in v1.

## Do not

Refactor these files in place. Add a new numbered patch instead.
