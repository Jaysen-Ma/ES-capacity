# Matched-budget ES vs. GRPO — 1.5B

**Status:** running at full budget since 2026-08-25 (50 steps, checkpoints every 10).
This document is the pre-registration and the runbook; measured numbers are marked as such.

The arm has now been **smoke-run end to end** (2 steps, Qwen2.5-1.5B, reduced batch, on
8x RTX 3060 12GB) to prove the pipeline executes: `actor/pg_clipfrac = 0.0` on every step
as required below, a nonzero reward and gradient once a rollout lands. That run fixed three
startup-fatal config errors — see [Checkpointing and resume](#checkpointing-and-resume) —
and surfaced two issues that affect interpretation of the results, not just execution:
[confound 5](#known-confounds) and
[the ES checkpoint format](#the-es-checkpoint-needs-converting-before-it-can-be-evaluated).
The timings in [Wall-clock expectation](#wall-clock-expectation) remain **unvalidated**;
the smoke config was deliberately far too small to extrapolate from.

## The question this run answers

> At an **identical generation and data budget**, does GRPO narrow the base model's pass@k
> ceiling more than ES does?

Both arms get 51 parameter updates, 256 prompts per step, 32 generations per prompt, a
2,048-token cap, the same 8,523 problems, the same prompt template, the same reward function,
and the same base checkpoint.

### What this run does *not* answer

- **Whether ES is under-trained.** Matching GRPO *down* to 51 steps makes both arms equally
  under-trained. It does not test whether ES's behaviour changes at the 500 iterations Qiu et
  al. recommend. That needs a separate, longer ES run.
- **Whether either arm beats the other at convergence.** This is a fixed-small-budget
  comparison, and should be described as one.
- **Anything about the base model's true ceiling.** See [Known confounds](#known-confounds).

## Budget arithmetic

The ES arm's real numbers, read from `es_trainer.py` and cross-checked against
`results/1.5b-sigma001-iter50/training_curves.csv` (51 rows):

| | ES (actual) | GRPO (this run) | note |
|---|---|---|---|
| Parameter updates | 51 | 51 | `--n-iterations 50` gives 51 (off-by-one in `fit()`) |
| Prompts/step | 256 (step 34: **75**) | 256 | ES `drop_last=False`; verl `drop_last=True` |
| Generations/prompt | 32 population | 32 rollouts | |
| Prompt-exposures | 12,875 | 13,056 | **+1.4%** — accepted, see below |
| Epochs | 1.5106 | 1.532 | |
| Generations | 412,000 | 417,792 | |
| Response cap | 2,048 | 2,048 | |

**The +1.4% exposure overshoot is accepted, not corrected.** ES's 34th step trained on a
75-prompt partial batch because its dataloader keeps the remainder; verl drops it. Forcing
verl to reproduce a ragged batch is more distortion than the 1.4% is worth. Report it.

## Arms

Four lines on every plot:

| arm | source | budget |
|---|---|---|
| **base** | `Qwen/Qwen2.5-1.5B` | — |
| **ES** | `zocrate/Qwen2.5-1.5B-ES-math` | 51 updates, 412k generations |
| **GRPO (matched)** | this run | 51 updates, 418k generations |
| **GRPO (published)** | SimpleRL-Zoo 1.5B checkpoint | ~400 updates, 819k generations, 8192 cap |

The published arm is **not** budget-comparable and is kept only as the link to the results
already in this repo. Label it as such on every plot.

## Stack

| component | version | why |
|---|---|---|
| trainer | **verl v0.7.1** | not SimpleRL's vendored verl 0.1 — see below |
| inference | **vllm==0.11.0** | matches the ES arm's engine exactly |
| venv | `/workspace/venvs/verl` | `/` has only ~4 GB free |

**Why not SimpleRL-Zoo's own code.** `hkust-nlp/simpleRL-reason` vendors verl 0.1 pinned to
`vllm<=0.6.3`, `transformers<4.48`, `ray==2.10.0` — an October 2024 stack. The ES arm ran on
vLLM 0.11.0. Since generation dominates both arms' wall-clock, running GRPO on a two-year-older
inference engine would make the wall-clock comparison a measurement of vLLM versions. We take
SimpleRL's *recipe* and run it on a modern trainer at the ES arm's inference version.

## Config

Values below are verified against `verl v0.7.1` config YAMLs, not HEAD (HEAD refactored
`fsdp_workers.py` away and renamed keys; v0.7.1 has not).

### Budget — matched to ES

```yaml
data.train_batch_size: 256
data.max_prompt_length: 2048        # NOT SimpleRL's 1024 — see below
data.max_response_length: 2048
data.filter_overlong_prompts: true
data.truncation: error
actor_rollout_ref.rollout.n: 32
trainer.total_training_steps: 51    # NOT 50
```

`max_prompt_length=2048` keeps all 8,523 problems. Measured with the Qwen2.5-1.5B tokenizer
under the ES template: max prompt is **1,682** tokens; 8 prompts (0.094%) exceed 1024, **zero**
exceed 2048. SimpleRL's 1024 would have dropped 8 problems the ES arm trained on.

### Exactly one gradient update per rollout batch

ES performs exactly one update per iteration (`update_weights_from_seeds`, once, after all 32
members are scored). To match:

```yaml
actor_rollout_ref.actor.ppo_epochs: 1
actor_rollout_ref.actor.ppo_mini_batch_size: 256   # == train_batch_size
```

`ppo_mini_batch_size` is denominated in **prompts** and multiplied by `rollout.n` internally, so
256 → 8,192 sequences = the whole batch = one update. Updates per step is
`ppo_epochs × (train_batch_size / ppo_mini_batch_size)`.

**Consequence worth knowing:** with one update per batch, `old_log_prob ≡ log_prob`, the
importance ratio is identically 1.0, and PPO clipping never binds. The objective collapses to
REINFORCE with a group-mean baseline. This *removes* a family of confounders — but only while
`ppo_epochs=1` holds. Assert it (below).

### Diversity knobs — all zeroed

pass@k *is* a diversity measure. Every knob that controls output diversity must be off, or the
result is partly a readout of that knob rather than of the algorithm.

```yaml
actor_rollout_ref.actor.use_kl_loss: false
actor_rollout_ref.actor.kl_loss_coef: 0.0
algorithm.use_kl_in_reward: false
algorithm.kl_ctrl.kl_coef: 0.0
actor_rollout_ref.actor.entropy_coeff: 0.0
actor_rollout_ref.actor.calculate_entropy: true    # log it, never optimize it
```

ES has no reference policy, no KL term, and no entropy bonus — its exploration is entirely
weight-space perturbation at σ=0.001. SimpleRL used `kl_loss_coef=1e-4` and
`entropy_coeff=1e-3`; **we deliberately deviate from their recipe here**, because a KL anchor to
the base model mechanically preserves pass@k and would make "GRPO preserved the ceiling"
tautological.

Zeroing both KL paths also means verl never constructs the reference-policy worker, which saves
a full forward pass per step (~8–15% of step time). It also means **verl will not log KL** — see
[Offline KL](#offline-kl).

### Optimization

```yaml
actor_rollout_ref.actor.optim.lr: 5e-7              # SimpleRL's default, not verl's 1e-6
actor_rollout_ref.actor.optim.lr_scheduler_type: constant   # ES's alpha is constant
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio: 0.0
actor_rollout_ref.actor.optim.weight_decay: 0.0     # override verl's 0.01
actor_rollout_ref.actor.optim.clip_grad: 1.0        # `grad_clip` is deprecated in v0.7.1
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-mean
algorithm.adv_estimator: grpo
algorithm.norm_adv_by_std_in_grpo: true
```

**`loss_agg_mode`.** ES fitness is the unweighted mean of 256 binary rewards — a 100-token
correct answer and a 1,000-token correct answer both count exactly 1.0. `token-mean` (verl's
default) would give the long one 10× the gradient weight. `seq-mean-token-mean` reproduces ES's
length-agnosticism.

**`norm_adv_by_std_in_grpo: true`** matches ES, which z-scores its population rewards
(`utils/reward_shaping.py`). The only difference is `ddof`: numpy population std vs torch sample
std, a factor of √(32/31) = 1.016. Negligible. Note this is the term Dr. GRPO
([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) argues is biased — we keep it *because*
it matches ES, not because it is correct.

**There is no principled mapping between ES's α=0.0005 and GRPO's lr.** ES's α multiplies a
unit-variance noise sum, giving a per-element parameter change of RMS α/√32 = 8.84e-5 per
iteration; a gradient learning rate is a different quantity entirely. This experiment is
therefore **"ES at its recommended settings vs GRPO at its recommended settings"**, and must be
worded that way.

### Generation — matched to ES

```yaml
actor_rollout_ref.rollout.temperature: 1.0
actor_rollout_ref.rollout.top_p: 1.0
actor_rollout_ref.rollout.top_k: -1
actor_rollout_ref.rollout.name: vllm
actor_rollout_ref.rollout.tensor_model_parallel_size: 1   # 8 independent engines, as ES
actor_rollout_ref.rollout.gpu_memory_utilization: 0.6
actor_rollout_ref.rollout.free_cache_engine: true
```

**Do not set `stop_token_ids`.** SimpleRL hard-codes `[14582, 16141, 31198]` ("Question",
"Answer", "Problem") for 1.5B in its `vllm_rollout.py`. ES sets none, and Qwen2.5-1.5B *base*
has `<|endoftext|>` (151643) as its only EOS — so ES rollouts run *past* `<|im_end|>` until EOS
or the 2,048 cap, and `last_boxed_only_string` grades the **last** `\boxed` in that text,
sometimes from hallucinated continuation. Inheriting SimpleRL's stop tokens would give cleaner
termination, shorter responses, and a materially different effective reward.

**Do not set `eos_token` to `<|im_end|>`.**

### System

```yaml
actor_rollout_ref.actor.strategy: fsdp2
actor_rollout_ref.model.use_remove_padding: true
actor_rollout_ref.model.enable_gradient_checkpointing: true
actor_rollout_ref.actor.use_dynamic_bsz: true
actor_rollout_ref.actor.ppo_max_token_len_per_gpu: 16384
trainer.n_gpus_per_node: 8
trainer.nnodes: 1
```

Measured on this box (8× RTX 4090, no NVLink, P2P disabled, NCCL host-staged SHM at 3.2–3.4
GB/s):

| config | tok/s | scaling efficiency |
|---|---|---|
| FSDP, micro-batch 1 | 4,765 | 7% |
| DDP + grad-accum 32 | 29,291 | 43% |
| **FSDP, micro-batch 16** | **45,969** | **67%** |
| *ideal 8-GPU* | *68,500* | *100%* |

FSDP re-all-gathers parameters every micro-batch, so gradient accumulation does **not** amortize
its comms — only a large micro-batch does. DDP amortizes well but OOMs at micro-batch 4 (49.2 GB
at mb=2) and cannot host colocated vLLM. Hence FSDP with a large micro-batch.

#### Deviation: this box is 8x RTX 3060 12GB, not 8x RTX 4090 48GB

Everything above was measured on 48GB cards. `run_grpo_matched.sh` now defaults to a 12GB
config, overridable per knob:

| env var | default | override key |
|---|---|---|
| `OFFLOAD` | `True` | `actor.fsdp_config.param_offload` / `.optimizer_offload` |
| `GPUMEM` | `0.85` | `rollout.gpu_memory_utilization` |
| `TOKGPU` | `4096` | `actor.ppo_max_token_len_per_gpu`, `rollout.log_prob_max_token_len_per_gpu` |
| `MAXBT` | `4096` | `rollout.max_num_batched_tokens` |
| `MAXSEQ` | `64` | `rollout.max_num_seqs` |

These are ported from the 12GB smoke that passed on 2026-08-25 and were re-validated with
`--cfg job --resolve`, but that smoke ran at `train_batch_size=8`, `rollout.n=4`,
`max_response_length=512`. **They are not yet validated at the matched budget** (256 x 32 x
2048); the first full-config step is the real test.

None of these change the optimization math. With `use_dynamic_bsz=true` and
`ppo_mini_batch_size == train_batch_size`, `ppo_max_token_len_per_gpu` sets only how a single
gradient update is split into accumulated micro-batches. `param_offload`/`optimizer_offload`
move state to host RAM between uses. The comparison against ES is preserved.

**But the throughput cost is severe and invalidates the wall-clock estimate below.**
`TOKGPU=4096` equals `max_prompt_length + max_response_length`, so a micro-batch is a single
full-length sequence -- the `micro-batch 1` row of the table above, 4,765 tok/s, not the
45,969 tok/s the 5h estimate is anchored on. At that rate the 8.93M tokens/step of
forward+backward alone is ~31 min, putting 51 steps near **26h rather than 4h49m**.

`TOKGPU` is bounded mainly by the vocabulary projection, not activations: logits for T tokens
cost `T x 151936 x 2` bytes in bf16 and are upcast to fp32 for `log_softmax`, so T=4096 is
~3.7GB of the 12GB on its own. Probe upward from 4096 before assuming the 26h figure --
recovering even 8192 roughly halves the gradient phase.

## Data preparation

Rebuild the parquet with the **ES template** — do not use SimpleRL's shipped `train.parquet`.

Three differences: ES puts the instruction in the **system** turn, SimpleRL in the **user** turn;
SimpleRL's text contains a literal f-string bug (`\boxed{{}}`, double braces); and verl v0.1
took the prompt verbatim from the parquet rather than applying a chat template.

Measured effect of using the ES template instead: `\boxed` emission rises **14.6% → 24.2%** and
mean response length falls **1,105 → 982 tokens** at identical accuracy — so it is ~11% cheaper
per step as well as being the correct match.

```
python scripts/build_matched_parquet.py \
  --out /workspace/data/lvl3to5_es_template/train.parquet
```

## Reward

Register ES's grader as verl's custom reward function:

```yaml
custom_reward_function.path: /workspace/es-capacity/scripts/es_reward_verl.py
custom_reward_function.name: compute_score
```

This wraps `es_at_scale.reward_function.math_grader.boxed_reward_fn` verbatim: binary {0.0, 1.0},
correctness only, **no format reward and no truncation penalty**, requiring a literal parseable
`\boxed{}`.

Do **not** use verl's `hf_math_verify`. Its `qwen_extract_answer` falls back through "the answer
is" → "final answer is" → **the last number in the string**, which makes GRPO's reward strictly
easier than ES's and removes the format headroom ES had to climb.

**Validation gate:** score ~200 base completions with both graders and record the disagreement
rate before the real run.

## Checkpointing and resume

Budget is **128 GB** on `/workspace`. A 1.5B checkpoint with optimizer state is ~25 GB
(bf16 params 3.1 + fp32 master 6.2 + Adam m,v 12.4 + hf_model 3.1).

```yaml
trainer.save_freq: 5
trainer.max_actor_ckpt_to_keep: 2          # NOT max_ckpt_to_keep -- see below
actor_rollout_ref.actor.checkpoint.save_contents: ['model','optimizer','extra','hf_model']
trainer.default_local_dir: /workspace/ckpt/grpo-1.5b-matched
```

**Three keys in earlier revisions of this document did not exist in verl v0.7.1**, and
each aborted the run during config composition, before any training:

| was | is |
|---|---|
| `trainer.max_ckpt_to_keep` | `trainer.max_actor_ckpt_to_keep` (retention is split per role) |
| `trainer.nccl_timeout` | `actor_rollout_ref.nccl_timeout` (`ppo_trainer.yaml:52`) |
| *(unset)* `data.val_files` | must be set — see below |

`data.val_files` is required even though validation is off: verl constructs a validation
dataset unconditionally, and `val_before_train=False` / `test_freq=-1` do not skip it. Left
unset it resolves to `~/data/rlhf/gsm8k/test.parquet` and the run dies with
`FileNotFoundError`. `run_grpo_matched.sh` points it at the train parquet; with validation
disabled it is never read.

Validate override keys against `/opt/verl/verl/trainer/config/ppo_trainer.yaml` (and the
`_generated_*.yaml` variants) rather than against this document.

→ ~50 GB peak, resume granularity 5 steps (~28 min of work lost worst case).

**Resume** is by `trainer.resume_mode`:

```
trainer.resume_mode=auto     # picks up the latest checkpoint in default_local_dir
trainer.resume_mode=disable  # force a fresh run
```

`save_contents` must include `optimizer` and `extra` or resume restarts the optimizer state and
the dataloader position, silently changing the experiment. `hf_model` is what the eval harness
consumes.

## Monitoring — abort criteria

Log every step. Watch these four:

| metric | expected | abort if |
|---|---|---|
| `frac_nonzero_adv_groups` | starts **30.9%**, climbs | flat by step 10 |
| `actor/pg_clipfrac` | **exactly 0.0** | ever non-zero → more than one update/batch |
| optimizer step count | 51 at the end | ≠ 51 |
| `response_length` mean | ~982, drifts | runaway growth toward the 2,048 cap |

Also log: entropy, reward mean/std, grad_norm, fraction of responses hitting the cap.

**The cold start is measured, not estimated.** A real step-1 batch (256 prompts × 32 rollouts,
T=1.0, graded with `boxed_reward_fn`) gave:

- pass@1 = **1.60%** (cross-validates ES's iteration-1 `reward_mean` of 0.0139 at T=0)
- **69.1%** of groups all-zero → contribute exactly zero gradient
- **30.9%** informative (79 of 256 groups)
- **47.5%** of total gradient magnitude comes from 46 prompts where exactly *one* of 32 rollouts
  succeeded, each handed advantage **+5.48** by std-normalization
- effective prompt sample size (Kish): **74.5 of 256**
- only **14.6%** of responses emit any `\boxed`; **37.1%** hit the 2,048 cap

Expect a flat, noisy reward curve for roughly the first 5–15 steps. **This is a result to
report, not a bug to fix** — it is the std-normalization pathology Dr. GRPO describes, biting
hardest exactly at cold start. It is also a cost ES does not pay: ES always has a fitness
ranking across its population, even when every rollout is wrong.

## Wall-clock expectation

Anchored on two numbers measured on this box: 8,192 completions generated in **87 s**, and
training throughput of **45,969 tok/s**.

| phase | est. |
|---|---|
| generation (8,192 rollouts) | ~77 s |
| forward+backward (8.93M tokens) | ~194 s |
| `old_log_prob` forward | ~48 s |
| weight sync, vLLM wake/sleep, optimizer | ~20 s |
| **per step** | **~340 s** |

**51 steps ≈ 4h49m**, range 4–7h. Compare ES's measured 3h22m35s.

> **Stale on current hardware.** These numbers assume 8x RTX 4090 48GB at micro-batch 16.
> Measured on this box's 12GB cards, 2026-08-25, step 1 at the full matched budget:

| phase | 4090 est. | **3060 measured** |
|---|---|---|
| generation (8,192 rollouts) | ~77 s | **487 s** |
| `old_log_prob` forward | ~48 s | **483 s** |
| forward+backward | ~194 s | **1,433 s** |
| weight sync | ~20 s | **6 s** |
| **per step** | **~340 s** | **2,413 s (40.2 min)** |
| **50 steps** | **~4h49m** | **~33.5 h** |

Aggregate throughput is **3,691 tok/s** against the 45,969 tok/s the 4090 estimate assumes --
part slower silicon, part the micro-batch-1 penalty from `TOKGPU=4096`.

**Raising `GPUMEM` is not the lever it looks like.** Generation is only 20% of the step;
`update_actor` is 59%. Halving generation time would save ~4 min/step, ~3h overall. The real
cost is the gradient phase, which is bounded by `TOKGPU`, and `TOKGPU` cannot be raised:
step 1 peaked at 7.86 GB allocated / 10.38 GB reserved of 11.63 GB, leaving under 1.3 GB
headroom, while doubling `TOKGPU` to 8192 would add ~1.3 GB to the logits tensor alone.
On this hardware ~33h is close to the floor for the matched budget.

Method check: ES's 230 s/iteration × 51 = 3h15m against a reported 3h22m — close enough to
trust the arithmetic.

**The decomposition is itself a finding.** GRPO's *generation* is ~3× faster than ES's (77 s vs
230 s for the identical 8,192 completions) because ES pays 32 weight-perturb-and-reload cycles
per iteration and generates in small per-member batches, while GRPO does one large batch off a
single policy with prefix caching across the 32 rollouts sharing each prompt. GRPO then spends
~240 s/step on gradient compute that ES never pays. Net: **GRPO ≈ 1.45× ES**, not the large
multiple one might assume from "backprop is expensive".

## The ES checkpoint needs converting before it can be evaluated

`ES-capacity` evaluates a model by pointing `scripts/run_eval.sh` at a HuggingFace
directory, which `math_eval.py` opens with `LLM(model=...)`. The ES trainer does not
produce one.

`save_self_weights_to_disk` does a plain `torch.save` over
`self.model_runner.model.named_parameters()` — that is **vLLM's internal parameter
layout**, not HuggingFace's. Verified against a checkpoint produced by this repo
(Qwen2.5-1.5B, 198 tensors):

```
model.layers.0.self_attn.qkv_proj.weight   (2048, 1536)   <- fused q+k+v
model.layers.0.mlp.gate_up_proj.weight     (17920, 1536)  <- fused gate+up
```

84 fused parameters, **zero** HF-style `q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj`,
and no `lm_head.weight` (Qwen2.5-1.5B ties embeddings, so vLLM never exposes it). A
single `pytorch_model.pth`, no `config.json`, no tokenizer files.

So there is a **required conversion step** between "ES run finishes" and "you can
evaluate it". The published artefacts — e.g. `zocrate/Qwen2.5-1.5B-ES-math` — are standard
HF directories, so the conversion is done as part of the normal workflow.

**The converter is deliberately not tracked in this repo.** Nothing here references
`qkv_proj` or `gate_up_proj`, and that is by choice, not an oversight — do not "fix" it by
committing one. The consequence to be aware of is only that a fresh clone does not carry
the step, so it has to be brought to the box separately. The spec below is recorded here so
the step is reconstructable if the untracked copy is ever lost.

The un-fusing itself is mechanical and well-defined for Qwen2:

- `qkv_proj` (2048) splits `1536 / 256 / 256` — 12 query heads and 2 KV heads at
  `head_dim=128` — into `q_proj`, `k_proj`, `v_proj`, for both `.weight` and `.bias`
- `gate_up_proj` (17920) splits `8960 / 8960` into `gate_proj`, `up_proj`
- `lm_head` is `embed_tokens` (tied); emit `tie_word_embeddings: true` rather than a
  duplicate tensor
- `config.json` / tokenizer files must be copied from the base model

Budget for this step when planning a run on a fresh box: the ES trainer's output is not
directly consumable by `run_eval.sh`, so "training finished" is not the same as "ready to
evaluate".

## Offline KL

Zeroing both KL paths means verl builds no reference policy and logs no KL. Measure it
afterwards:

```
python scripts/measure_kl.py --base Qwen/Qwen2.5-1.5B --model <ckpt> --n 512
```

**Why this matters more than it looks.** The ES-at-scale paper's own Table 2 reports mean KL
from base at *exactly this project's ES hyperparameters* (σ=0.001, α=0.0005) on Qwen2.5-7B:

| arm | KL from base |
|---|---|
| ES (σ=0.001) | **0.274** |
| GRPO (β=0.0) | 0.861 |
| GRPO (β=0.0167) | 1.591 |

ES sits 3–6× closer to base than GRPO at any β they tested. So "ES preserves the pass@k ceiling"
has an unexcluded null hypothesis: **ES barely moved the weights.** That is the mirror image of
the objection one would raise against a KL-regularized GRPO arm.

Report achieved KL for every arm and plot **reward against KL**, not single points. Without it
the headline claim is not falsifiable.

## Known confounds

Carried into this run and **not** fixed by budget matching:

1. **Training temperature is irreconcilable.** ES trained 100% greedily
   (`train_temperature = 0.0`, hardcoded, all 412,000 rollouts). GRPO cannot run at T=0 —
   identical rollouts give zero group variance and zero gradient. This is a structural
   asymmetry between the methods, not a knob to equalize. Name it; do not paper over it.
   pass@k is a property of the T>0 distribution that ES was never optimized against.

2. **The base-model evaluation is zero-shot.** `scripts/run_eval.sh` never passes `--num_shots`
   and `math_eval.py` defaults it to 0, so the base is scored inside a ChatML frame it was never
   trained to follow (MATH-500 pass@1 = 5.4%, vs ~25–30% for this model under few-shot CoT).
   Yue et al. use few-shot for base models specifically to avoid this. **Parked by decision** —
   it does not affect the ES-vs-GRPO contrast, since both arms are compared to each other under
   identical settings. It *does* affect every claim about the base ceiling.

3. **Information asymmetry at equal generations.** ES's 8,192 rollouts collapse into 32 scalars
   (one fitness per population member); GRPO's 8,192 yield 8,192 per-sequence advantages. At
   equal generations GRPO extracts far more estimator information. The generation denominator
   structurally favours GRPO; the wall-clock denominator favours ES. Report both.

4. **Single seed.** One run per arm, and AIME24 is 30 questions. A pass@k crossover at n=30 with
   one seed is suggestive, not conclusive. Bootstrap CIs over questions at minimum.

5. **The training reward and the reported metric are different functions, and the gap is
   asymmetric between the arms.** This is the one confound here that is not merely a
   limitation — it cuts directly at the ES-vs-RL contrast.

   Three different answer-extraction rules are in play. The prompt template is shared;
   the extraction is not.

   | stage | extraction | no `\boxed{}` present | equivalence |
   |---|---|---|---|
   | **ES training** (`boxed_reward_fn`) | last `\boxed{}` **only** | **reward 0.0** | mathd / sympy / math-verify |
   | **GRPO matched** (`es_reward_verl.py`) | same — wraps ES verbatim | **reward 0.0** | same as ES |
   | **SimpleRL-Zoo training** (`hf_math_verify`) | boxed -> `he answer is` -> `final answer is` -> **last number in string** | scored anyway | math-verify |
   | **pass@k eval** (`math_eval/parser.py`) | *identical fallback chain to SimpleRL's* | scored anyway | `math_equal`, `include_percentage=True` |

   The eval harness and SimpleRL-Zoo both use the Qwen2.5-Math eval toolkit parser — the
   `extract_answer` logic is the same. So **the published RL arm was trained against
   essentially the extraction rule it is scored with, and the ES arm was not.**

   Measured on the two graders as installed (7 hand-built cases, 5 disagree):

   | response | ground truth | ES training reward | pass@k eval |
   |---|---|---|---|
   | `The answer is 42.` | 42 | 0.0 | **1.0** |
   | `...so the final answer is 42` | 42 | 0.0 | **1.0** |
   | `After simplifying we get 42` | 42 | 0.0 | **1.0** |
   | `I think it's 7 or maybe 42` | 42 | 0.0 | **1.0** |
   | `\boxed{0.5}` | 50 | 0.0 | **1.0** (`include_percentage`) |
   | `Thus \boxed{42}.` | 42 | 1.0 | 1.0 |
   | `\boxed{42}` | 42 | 1.0 | 1.0 |

   Note the last-number fallback scores a hedged non-answer, and `include_percentage=True`
   in `math_equal` accepts `reference/100`, `reference` and `reference*100` alike.

   **Which way it cuts.** ES was trained to emit `\boxed{}` — this document records
   emission rising 14.6% -> 24.2% under the ES template. The eval awards credit without
   it, so that format gain is largely invisible in the reported pass@k, while SimpleRL
   never needed it. Under a strict-boxed eval the ES arm would likely look *better*
   relative to RL than it currently does. The comparison between arms stays internally
   fair — every arm is scored under the same eval — but "ES preserves the ceiling" is
   being measured through an extraction path ES was never rewarded for.

   The same reasoning already appears in this document, applied to the *matched* GRPO arm:
   verl's `hf_math_verify` was rejected because it "would hand GRPO an easier reward and
   remove the format headroom ES had to climb." That argument was never extended to the
   eval side, or to the published SimpleRL checkpoint in the headline 7B results.

   **The existing validation gate does not cover this.** `scripts/check_grader_agreement.py`
   compares ES's grader against verl's `hf_math_verify` — i.e. two *training* graders. It
   says nothing about ES-training vs pass@k-eval, which is the pair that matters here.

   **Secondary: the grader has an unpinned version.** `math-verify` reached `/venv/train`
   transitively through verl's unversioned `math` extra, and it *is* in the ES reward path
   — `es_trainer.py` calls the reward via `functools.partial(reward_function)` with no
   `fast` argument, so `fast=False` and `is_latex_equal` (math-verify) runs as the final
   arm of the correctness check. Versions disagree: 0.6.0 grades `\boxed{50\%}` against
   `0.5` as correct, 0.9.0 does not. It is now pinned at 0.9.0 in the image, but **which
   version produced the already-published ES results is not recorded anywhere.** Relatedly,
   `reward_function_timeout` (default 10s) converts a slow grade into reward 0.0, and
   math-verify is the slow path — so grader version also perturbs the timeout rate.

   **What would resolve it:** re-score the existing completion dumps under ES's strict
   grader and report pass@k both ways. This needs no retraining — `run_eval.sh` already
   writes per-sample outputs — and it converts an unquantified confound into a number.

## Runbook

```bash
# 0. free the GPUs and stop the image's Ray (verl starts its own head)
supervisorctl stop vllm model-ui ray

# 1. environment (already built)
source /venv/train/bin/activate

# 2. data
python scripts/build_matched_parquet.py --out /workspace/data/lvl3to5_es_template/train.parquet

# 3. grader agreement gate
python scripts/check_grader_agreement.py --n 200

# 4. smoke test — 3 steps, ~20 min
bash scripts/run_grpo_matched.sh --smoke

# 5. full run — ~5h
bash scripts/run_grpo_matched.sh
```

### Smoke-test acceptance criteria

Do not start the long run until all of these hold:

Status after step 1 of the full run (2026-08-25, `logs/grpo_full.log`), which is a
stronger check than the smoke since it runs the real budget:

| criterion | expected | step 1 | |
|---|---|---|---|
| `actor/pg_clipfrac` | 0.0 | 0.0 | pass |
| `actor/kl_loss` | 0.0 | 0.0 | pass |
| mean response length | ~982 | 979.1 | pass |
| non-zero rewards appear | yes | 0.0143 | pass |
| no OOM at full budget | - | 7.86/11.63 GB | pass |

`response_length/mean = 979` against a pre-registered ~982 is the notable one: the
generation distribution matches the ES arm's despite a different trainer and sampler.
`response_length/clip_ratio = 0.287` -- 29% of responses hit the 2048 cap and therefore
score 0.0 under ES's strict `\boxed{}` grader, which is the intended behaviour, not a bug.

- [ ] flash-attn imports and the varlen path is active (`use_remove_padding` effective).
      Without it you pay 25.2M padded tokens/step instead of 10.0M — 2.5× the compute,
      silently, with no error.
- [ ] `actor/pg_clipfrac == 0.0` on every step
- [ ] no reference-policy worker in the Ray dashboard
- [ ] non-zero rewards appear; `frac_nonzero_adv_groups` ≈ 0.30
- [ ] mean response length ≈ 982 tokens
- [ ] no OOM at micro-batch 16 with vLLM colocated
- [ ] checkpoint writes, and `resume_mode=auto` reloads it
