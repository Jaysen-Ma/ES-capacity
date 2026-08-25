# ES-capacity Docker image

One image that runs all four repos in this project: ES training (`es-at-scale`), GRPO training
(`verl`), pass@k evaluation (`limit-of-RLVR`), and `simpleRL-reason` for reference — plus the
`gh` and `claude` CLIs.

## Layout

| path | contents |
|---|---|
| `/venv/train` | torch 2.8, vllm 0.11.0, transformers 4.57.6, verl v0.7.1, flash-attn 2.8.3 — serves **both** ES and GRPO training |
| `/venv/eval` | torch 2.4, vllm 0.6.3 (pinned `==`, not `<=`), transformers <4.48, setuptools -- the `math_eval` pass@k harness |
| `/opt/verl` | verl v0.7.1, installed editable into `/venv/train` |
| `/opt/es-capacity/on_start.sh` | clones/pulls the four forks to `$WORKSPACE/repos` at boot, installs the two repo-vendored packages, and runs a preflight import check |

Two packages live in the repos, not the image, and are installed by `on_start.sh` at boot:
`es_at_scale` into `/venv/train` and the vendored `latex2sympy2` into `/venv/eval`. Both
are required — without them the ES arm and the pass@k harness respectively fail at import.
Boot prints a preflight block; if any line says `**BROKEN**`, stop and fix it before
starting a run.

Repos are **not** baked in — they are actively developed. Only the slow dependency stacks live
in the image.

## Build requirements

- **x86_64 host.** This is the CPU architecture (Intel/AMD), not an OS — Windows, Linux, and
  Intel Macs all qualify. **Apple Silicon (M1–M4) does not**; it is ARM64 and would need
  `--platform linux/amd64` emulation, which is impractically slow for a ~31 GB CUDA image.
- **~80 GB free disk.** This is the binding constraint, not RAM.
- **~8 GB RAM.** Nothing is compiled — flash-attn comes from a prebuilt wheel.
- Docker with BuildKit.

```bash
docker build -f docker/Dockerfile -t ghcr.io/jaysen-ma/es-capacity:latest .
echo "$GHCR_PAT" | docker login ghcr.io -u jaysen-ma --password-stdin
docker push ghcr.io/jaysen-ma/es-capacity:latest
```

Then launch a Vast instance from `ghcr.io/jaysen-ma/es-capacity:latest`, with `on_start.sh` as
the onstart script.

## Why the previous version could not build

Four independent failures, each fatal on its own. All are fixed here; the notes exist so they
are not reintroduced.

1. **The base image tag never existed.** The old file used
   `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310`. vastai publishes
   `...-ubuntu24.04-py312-2026-08-21` — note 24.04, not 22.04, and a mandatory date suffix with
   no undated variant. This fails on the `FROM` line, before anything else runs.

2. **flash-attn was compiled from source** via
   `MAX_JOBS=4 pip install --no-build-isolation flash-attn` with
   `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` — a multi-hour, many-GB-RAM build across four
   architectures. On a GitHub Actions runner this presents as *"the hosted runner lost
   communication with the server"*, which is the OOM/disk-exhaustion signature, not a network
   fault. Fixed by installing Dao-AILab's prebuilt wheel; **no nvcc runs during this build.**

3. **Pre-release pins were not allowed.** `hydra-core==1.4.0.dev1` and `omegaconf==2.4.0.dev3`
   are dev releases; uv refuses them without `--prerelease=allow`.

4. **Three venvs did not fit.** The old file assumed ES, verl, and eval all conflict. They do
   not — verl v0.7.1 leaves `transformers` unpinned (`setup.py:41`) and accepts
   `vllm>=0.8.5,<=0.12.0` (`setup.py:52`), both satisfied by es-at-scale's
   `transformers==4.57.6` / `vllm==0.11.0`. Merging them cuts ~19 GB.

## Pinning rationale

**`vllm==0.11.0` is load-bearing, not incidental.** The ES arm was trained on it. The
ES-capacity experiment compares ES and GRPO wall-clock on the same box, and generation dominates
both arms — so if the two arms ran different vLLM versions, the comparison would measure vLLM
rather than the algorithms. See `docs/matched-experiment.md`.

**`/venv/eval` is frozen deliberately.** The published pass@k numbers in this repo were produced
under that stack. Upgrading it would break comparability with results already committed.

**flash-attn GPU coverage — verified, not assumed.** The ABI note below is about
*linking*; this is about whether the kernels run at all. The wheel ships native cubins for
**sm_80, sm_90, sm_100, sm_120** and **no PTX whatsoever**, so there is no JIT fallback to
paper over a missing architecture. Neither the 4090 (sm_89) nor the 3060 (sm_86) has its
own cubin. They work because of CUDA's minor-version rule — a `sm_X.y` cubin runs on any
`sm_X.z` with `z >= y` — so both are served by the sm_80 cubin. H100 (sm_90) is native.
Confirmed empirically by running a real `flash_attn_func` call on sm_86 hardware, rather
than trusting the rule. Expect untuned (working, but not Ada-optimal) performance on 4090.
If a future target is a new *major* compute capability with no cubin listed above, this
wheel will fail at kernel launch with no warning at import time.

**flash-attn ABI.** The wheel's `cxx11abi` tag must match torch's `_GLIBCXX_USE_CXX11_ABI`.
The rule is **version-based, not index-based**: every torch >=2.7 Linux wheel sets the flag on
after PyTorch's manylinux_2_28 migration, and the default PyPI wheel for 2.8.0 is itself the
cu128 build. `vllm==0.11.0` pulls torch 2.8.0+cu128 from stock PyPI -- no `index-url` is set
anywhere in this repo -- hence `FA_ABI=TRUE`. There is no ABI=0 torch 2.8 wheel to reason from.
The Dockerfile asserts at build time rather than assuming, so a mismatch fails loudly instead
of producing an image that dies at `import flash_attn`.

## Grader versioning

**The reward function has a version, and it was not pinned.** `math-verify` reached
`/venv/train` transitively through verl's `math` extra, which declares it unversioned
(`verl/setup.py`: `MATH_REQUIRES = ["math-verify"]`). That made the training reward a
function of the build date. It is now pinned to `0.9.0`.

This is not hypothetical. The two venvs currently disagree:

| response | ground truth | math-verify 0.6.0 | math-verify 0.9.0 |
|---|---|---|---|
| `\boxed{50\%}` | `0.5` | correct | **not correct** |

`/venv/eval` is frozen at `0.6.0` (deliberately, to preserve comparability with published
pass@k numbers); `/venv/train` is now pinned at `0.9.0`. **They do not match, by
construction.** Aligning them is a research decision, not a build fix — it requires
deciding which grader version the already-published ES results should be attributed to,
which is not currently recorded anywhere.

### The larger issue: three different graders

Version drift is the smaller half. The extraction methods differ outright:

| stage | extraction | no `\boxed{}` present | equivalence |
|---|---|---|---|
| **ES training** (`boxed_reward_fn`) | last `\boxed{}` **only** | **reward 0.0** | mathd / sympy / math-verify |
| **GRPO matched** (`es_reward_verl.py`) | same — wraps ES verbatim | **reward 0.0** | same as ES |
| **SimpleRL-Zoo training** (`hf_math_verify`) | boxed → `he answer is` → `final answer is` → **last number in string** | scored anyway | math-verify |
| **pass@k eval** (`math_eval/parser.py`) | *identical chain to SimpleRL's* | scored anyway | `math_equal`, `include_percentage=True` |

The eval harness and SimpleRL-Zoo share the Qwen2.5-Math eval toolkit parser, so **the
published RL arm was trained against essentially the same extraction rule it is scored
with, and the ES arm was not.** `docs/matched-experiment.md` is careful to give the
*matched* GRPO arm ES's strict grader so GRPO does not get an easier reward — but that
reasoning was never applied to the eval side, or to the published SimpleRL checkpoint
that appears in the headline results.

Measured on the two graders as installed (7 hand-built cases, 5 disagree):

| response | gt | ES training | pass@k eval |
|---|---|---|---|
| `The answer is 42.` | 42 | 0.0 | **1.0** |
| `...so the final answer is 42` | 42 | 0.0 | **1.0** |
| `After simplifying we get 42` | 42 | 0.0 | **1.0** |
| `I think it's 7 or maybe 42` | 42 | 0.0 | **1.0** |
| `\boxed{0.5}` | 50 | 0.0 | **1.0** (`include_percentage`) |
| `Thus \boxed{42}.` | 42 | 1.0 | 1.0 |
| `\boxed{42}` | 42 | 1.0 | 1.0 |

Nothing here is necessarily wrong — all arms are scored under the same eval, so the
comparison between them stays internally fair. But the training objective and the
reported metric are not the same function, the gap is asymmetric between the ES and RL
arms, and that belongs in the writeup rather than in a build file.

## Operational pitfalls

Things that will not fail the build, but will cost you a session.

**`on_start.sh` moves ES-capacity onto `main` at every boot.** It syncs that repo to
`${ES_CAPACITY_REF:-main}`. If you are working on a branch, a reboot silently checks out
`main` and your work disappears from the working tree — the commits survive on the branch,
but nothing tells you the tree moved. Export `ES_CAPACITY_REF=<your-branch>` in the
instance's environment, or merge before rebooting.

**A dirty working tree makes the boot-time pull fail, quietly.** `sync_repo` runs
`git pull --ff-only` and swallows failure with a `WARN` line, so a rebooted instance
happily runs stale code. Treat *commit and push before teardown* as a hard rule: these
repos are cloned fresh at boot, so anything unpushed is gone when the instance dies.

**ES engine startup is ~8x longer than it needs to be.** `es_trainer.py` gives every
engine its own `VLLM_CACHE_ROOT`, so none of the 8 reuse the first one's `torch.compile`
artefacts, and they are constructed serially. Measured ~15 min to first iteration on
8x RTX 3060; proportionally less on faster cards, but still 8x redundant compilation. The
per-engine isolation was a workaround for vLLM 0.26.0 and `SETUP_NOTES.md` records that it
was never re-tested under the pinned 0.11.0.

**Set `HF_HUB_OFFLINE=1` once models are cached.** Otherwise every engine start blocks on
10s `ReadTimeoutError`s doing `HEAD config.json` for a model already on disk.

**verl v0.7.1 routes rollouts through `AgentLoopWorker` + `vLLMHttpServer`**, which is
newer than what `docs/matched-experiment.md` describes. Metrics come out as expected
(`num_turns: 2.0`, single-turn), but that document's acceptance criterion "no
reference-policy worker in the Ray dashboard" was written against the older worker
implementation and should be re-checked against the dashboard on the first real run rather
than assumed.

## Build-time knobs

```bash
docker build -f docker/Dockerfile \
  --build-arg VERL_REF=v0.7.1 \
  --build-arg FA_VERSION=2.8.3.post1 \
  --build-arg FA_ABI=TRUE \
  --build-arg BASE_IMAGE=vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312-2026-08-21 \
  -t ghcr.io/jaysen-ma/es-capacity:latest .
```

## If you build on GitHub Actions

A standard runner has ~14 GB free, which is not enough for ~31 GB of venvs. Reclaim ~25 GB
first:

```yaml
- name: Free disk space
  run: |
    sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android /opt/hostedtoolcache
    sudo apt-get clean
    df -h /
```

That yields ~39 GB — enough, but not comfortable. A larger runner, or building on a machine you
control, is the safer path.

## Fixed after the first end-to-end build

The four above were found by inspection. Actually building the image surfaced three more, none
of them visible without running it:

5. **`FA_ABI` defaulted to `FALSE`.** Every torch >=2.7 Linux wheel sets the cxx11 ABI flag on,
   so the assert tripped and the build stopped at step 4/9. Now `TRUE`. See *flash-attn ABI*.

6. **`/venv/eval` had no `setuptools`.** `uv venv` does not seed it and vllm imports it at
   runtime, so the venv built clean and then died at `import vllm` -- at eval time, not build
   time.

7. **`vllm<=0.6.3` is not a freeze.** With no lower bound the resolver walked down to
   vllm 0.5.0.post1 / torch 2.3.0. Now pinned `==0.6.3`.

Separately: `.gitattributes` forces LF for `*.sh`, and the image strips CR from `on_start.sh` at
build time. A Windows checkout (`core.autocrlf=true`, the Git-for-Windows default) otherwise
bakes in a CRLF boot hook that Linux rejects with `bad interpreter: bash^M` -- the image builds
green and the instance then boots with no repos cloned.
