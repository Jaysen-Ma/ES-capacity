# ES-capacity Docker image

One image that runs all four repos in this project: ES training (`es-at-scale`), GRPO training
(`verl`), pass@k evaluation (`limit-of-RLVR`), and `simpleRL-reason` for reference — plus the
`gh` and `claude` CLIs.

## Layout

| path | contents |
|---|---|
| `/venv/train` | torch 2.8, vllm 0.11.0, transformers 4.57.6, verl v0.7.1, flash-attn 2.8.3 — serves **both** ES and GRPO training |
| `/venv/eval` | torch 2.4, vllm ≤0.6.3, transformers <4.48 — the `math_eval` pass@k harness |
| `/opt/verl` | verl v0.7.1, installed editable into `/venv/train` |
| `/opt/es-capacity/on_start.sh` | clones/pulls the four forks to `$WORKSPACE/repos` at boot |

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

**flash-attn ABI.** The wheel's `cxx11abi` tag must match torch's `_GLIBCXX_USE_CXX11_ABI`.
torch 2.8 PyPI wheels are built with it off, hence `FA_ABI=FALSE`. The Dockerfile asserts this
at build time rather than assuming it, so an upstream change fails loudly instead of producing
an image that dies at `import flash_attn`.

## Build-time knobs

```bash
docker build -f docker/Dockerfile \
  --build-arg VERL_REF=v0.7.1 \
  --build-arg FA_VERSION=2.8.3.post1 \
  --build-arg FA_ABI=FALSE \
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
