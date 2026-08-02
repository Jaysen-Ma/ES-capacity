# ES-capacity

**Does evolution-strategy post-training expand LLM reasoning capacity beyond the base model?**

This project generalises the pass@k capacity analysis of Yue et al. (NeurIPS 2025) from reinforcement learning with verifiable rewards (RLVR) to evolution-strategy (ES) post-training, using two recent ICML 2026 ES methods.

## Papers

| Role | Paper | Link |
|------|-------|------|
| Capacity analysis (pass@k) | Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* | [arXiv:2504.13837](https://arxiv.org/abs/2504.13837) |
| Full-parameter ES | Qiu et al., *Evolution Strategies at Scale* | [arXiv:2509.24372](https://arxiv.org/abs/2509.24372) |
| Low-rank ES (EGGROLL) | Sarkar et al., *Evolution Strategies at the Hyperscale* | [arXiv:2511.16652](https://arxiv.org/abs/2511.16652) |

## Method (high level)

1. Start from a base LLM.
2. Post-train with Qiu-style full-parameter ES and Sarkar EGGROLL on verifiable reasoning tasks.
3. Measure reasoning coverage with pass@k at large \(k\) for base vs ES-trained models.
4. Compare capacity curves and sampling-efficiency gap \(\Delta_{\mathrm{SE}}\) (Yue et al.).

## Package map

```
es_capacity/
  config.py          ExperimentConfig placeholders
  data.py            Prompts / verifiable tasks
  model.py           Load base LLM; apply / restore param noise
  reward.py          Verifiable reward R(x, y) ∈ {0, 1}
  pipeline.py        train ES → eval pass@k → compare
  es/
    base.py          Shared ES trainer loop skeleton
    qiu.py           Qiu full-parameter ES
    eggroll.py       Sarkar EGGROLL low-rank perturbations
  eval/
    passk.py         Unbiased pass@k estimation
    capacity.py      Base vs ES curves + Δ_SE
  analysis/
    coverage.py      Solvable-set overlap / narrowing
    perplexity.py    "Paths already in base" probes
```

## Status

**Skeleton only.** Modules are pseudo-Python stubs (`NotImplementedError`) that name components; nothing is runnable yet.
