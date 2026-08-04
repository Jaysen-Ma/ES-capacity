# Vendored: Qiu et al. ES-at-Scale (full-rank ES)

| Field | Value |
|-------|-------|
| Upstream | https://github.com/VsonicV/es-at-scale (fork: Jaysen-Ma/es-at-scale) |
| Local clone | `/home/zotar/jupyterlab/es-at-scale` |
| Commit | `574a9d134da1ffce2a8bb812019899e5c96b588a` |
| Paper | arXiv:2509.24372 |

## Files copied

- `es_at_scale/` package (`train.py`, `trainer/es_trainer.py`, `utils/worker_extension.py`, reward + template modules)
- `setup.py`, `README.md`, `LICENSE.txt`

Training datasets remain in the sibling clone (`../es-at-scale/datasets/`) to avoid duplicating large files.

## Role in v1

Interface-only / v2 drop-in. EGGROLL is the v1 ES arm. `es_capacity.posttrain.qiu` wraps the full-rank Perturber (`torch.randn` in-place via `worker_extension_cls`) without a full training loop in v1.
