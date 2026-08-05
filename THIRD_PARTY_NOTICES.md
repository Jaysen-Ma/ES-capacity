# Third-party notices

Our own code (`es_capacity/`, `scripts/`, `configs/`) is MIT — see [LICENSE](LICENSE).

Everything under `third_party/` is **someone else's code under someone else's terms**. It is
vendored so the analysis is reproducible from one clone, not relicensed. Two of the three trees
are more restrictive than MIT, and one has no licence at all. Check the relevant row before
reusing anything from there.

| Tree | Upstream | Commit | Licence |
|------|----------|--------|---------|
| `third_party/yue_math/` | [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR) | `79c348f4` | **None stated** — all rights reserved by default |
| `third_party/eggroll/` | [ESHyperscale/eggroll-vllm](https://github.com/ESHyperscale/eggroll-vllm) | `bcc215e8` | **GPL-3.0** (copyleft) — `third_party/eggroll/LICENSE` |
| `third_party/qiu_es/` | [VsonicV/es-at-scale](https://github.com/VsonicV/es-at-scale) | `574a9d13` | **Academic Public License — non-commercial only** — `third_party/qiu_es/LICENSE.txt` |

Each tree's `VENDOR.md` records its upstream commit, a recipe to re-derive it, and every patch
we apply on top. Divergence from upstream is always a numbered patch under `patches/`, never a
silent in-place edit — which also serves as the dated record of changes that GPL-3.0 requires.

## What each one means in practice

**`yue_math` — no licence.** Upstream ships no `LICENSE` and states none in its README, so
there is no explicit grant to redistribute it. We believe this is an oversight — the repo is
published so others can reproduce the paper's pass@k analysis, which is precisely this
project's purpose — and we have asked the authors to add one. Our use is academic,
non-commercial and attributed, and the tree is kept byte-identical to upstream (modulo recorded
patches) so it can be removed cleanly if the authors prefer. **This code is not ours to
relicense; the repo's MIT licence does not extend to it.**

**`eggroll` — GPL-3.0.** Strong copyleft: a distributed derivative of these files must itself
be GPL-3.0. To keep our own code MIT, `es_capacity/` does not port or adapt these sources. The
ES algorithm implemented in `es_capacity/posttrain/` works from the paper (arXiv:2511.16652) —
algorithms are not copyrightable, expression is — while this tree is retained as reference and
as the upstream launch path.

**`qiu_es` — non-commercial only.** Free use is limited to academic teaching and research,
non-profit research, and personal non-profit purposes. **Commercial use requires a separate
licence from Cognizant Technology Solutions Corp**, including a commercial entity participating
in a research project. v1 use is academic and nothing in `es_capacity/` imports this tree.
Flagged because it is the constraint most likely to be missed later: if this work moves into a
commercial setting, this directory must be resolved first.

## Papers

| Role | Paper | arXiv |
|------|-------|-------|
| Capacity analysis | Yue et al. | 2504.13837 |
| Low-rank ES (EGGROLL) | Sarkar et al. | 2511.16652 |
| Full-parameter ES | Qiu et al. | 2509.24372 |
| GRPO baseline (SimpleRL-Zoo) | Zeng et al. | 2503.18892 |

Datasets under `data/` carry their own provenance and SHA256 in their respective directories.
