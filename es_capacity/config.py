"""Experiment and local machine configuration.

Local paths live in ``config.toml`` (gitignored). Copy ``config.sample.toml``
and edit. Override the path with ``$ES_CAPACITY_CONFIG`` if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

if sys.version_info < (3, 11):
    raise RuntimeError("es_capacity.config requires Python 3.11+ (tomllib)")

import tomllib

ESMethod = Literal["qiu", "eggroll"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_NAME = "config.toml"
SAMPLE_CONFIG_NAME = "config.sample.toml"


@dataclass
class ESConfig:
    """Shared ES hyperparameters (algorithm outline only)."""

    method: ESMethod = "qiu"
    population_size: int = 30
    noise_scale: float = 0.001  # σ
    learning_rate: float = 5e-4  # α
    num_iterations: int = 0
    # EGGROLL-only: rank of per-member low-rank perturbation A B^T
    rank: int = 1


@dataclass
class EvalConfig:
    """pass@k evaluation settings (Yue et al.)."""

    ks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256])
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 4096
    # Proxy upper bound for Δ_SE vs base pass@k_max
    k_max: int = 256


@dataclass
class ExperimentConfig:
    """Top-level experiment config for capacity comparison."""

    base_model: str = ""
    dataset: str = ""
    es: ESConfig = field(default_factory=ESConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


@dataclass(frozen=True)
class ModelSpec:
    """One local HF checkpoint + OpenAI served name."""

    key: str
    path: Path
    served_name: str


@dataclass(frozen=True)
class LocalConfig:
    """Parsed ``config.toml`` for paths, models, vLLM, and eval defaults."""

    path: Path
    models_dir: Path
    venv: Path | None
    models: dict[str, ModelSpec]
    vllm: dict[str, Any]
    eval: dict[str, Any]

    def model(self, key: str) -> ModelSpec:
        if key not in self.models:
            known = ", ".join(sorted(self.models)) or "(none)"
            raise KeyError(f"Unknown model key {key!r}; known: {known}")
        return self.models[key]

    def model_paths(self, keys: list[str] | None = None) -> list[str]:
        ordered = keys if keys is not None else list(self.models.keys())
        return [str(self.model(k).path) for k in ordered]


def config_path(explicit: str | Path | None = None) -> Path:
    """Resolve which config file to load."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ES_CAPACITY_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / DEFAULT_CONFIG_NAME).resolve()


def _resolve_model_path(models_dir: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (models_dir / p).resolve()


def load_local_config(explicit: str | Path | None = None, *, required: bool = True) -> LocalConfig:
    """Load local machine config from TOML."""
    path = config_path(explicit)
    if not path.is_file():
        sample = REPO_ROOT / SAMPLE_CONFIG_NAME
        msg = (
            f"Missing config file: {path}\n"
            f"Copy the sample and edit paths:\n"
            f"  cp {sample} {REPO_ROOT / DEFAULT_CONFIG_NAME}"
        )
        if required:
            raise FileNotFoundError(msg)
        raise FileNotFoundError(msg)

    with path.open("rb") as f:
        raw = tomllib.load(f)

    paths = raw.get("paths") or {}
    models_dir_raw = str(paths.get("models_dir", "")).strip()
    if not models_dir_raw:
        raise ValueError(f"{path}: [paths].models_dir is required")
    models_dir = Path(models_dir_raw).expanduser().resolve()

    venv_raw = paths.get("venv")
    venv = Path(str(venv_raw)).expanduser().resolve() if venv_raw else None

    models_raw = raw.get("models") or {}
    models: dict[str, ModelSpec] = {}
    for key, spec in models_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: [models.{key}] must be a table")
        rel = spec.get("path")
        served = spec.get("served_name")
        if not rel or not served:
            raise ValueError(f"{path}: [models.{key}] needs path and served_name")
        models[str(key)] = ModelSpec(
            key=str(key),
            path=_resolve_model_path(models_dir, str(rel)),
            served_name=str(served),
        )

    return LocalConfig(
        path=path,
        models_dir=models_dir,
        venv=venv,
        models=models,
        vllm=dict(raw.get("vllm") or {}),
        eval=dict(raw.get("eval") or {}),
    )


def dump_config_value(key: str, explicit: str | Path | None = None) -> str:
    """Print a resolved config field (used by shell scripts)."""
    cfg = load_local_config(explicit, required=True)
    if key == "config_path":
        return str(cfg.path)
    if key == "models_dir":
        return str(cfg.models_dir)
    if key == "venv":
        if cfg.venv is None:
            raise KeyError("paths.venv is not set")
        return str(cfg.venv)
    if key.startswith("model."):
        parts = key.split(".")
        if len(parts) != 3:
            raise KeyError(f"Expected model.<name>.path|served_name, got {key!r}")
        _, name, field_name = parts
        spec = cfg.model(name)
        if field_name == "path":
            return str(spec.path)
        if field_name == "served_name":
            return spec.served_name
        raise KeyError(f"Unknown model field {field_name!r}")
    if key.startswith("vllm."):
        field_name = key.split(".", 1)[1]
        if field_name not in cfg.vllm:
            raise KeyError(f"Unknown vllm field {field_name!r}")
        return str(cfg.vllm[field_name])
    if key.startswith("eval."):
        field_name = key.split(".", 1)[1]
        if field_name not in cfg.eval:
            raise KeyError(f"Unknown eval field {field_name!r}")
        val = cfg.eval[field_name]
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
        return str(val)
    raise KeyError(f"Unknown config key {key!r}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Print a value from config.toml")
    p.add_argument("key", help="e.g. model.base.path, vllm.port, eval.output_dir")
    p.add_argument("--config", default=None, help="Path to config.toml")
    args = p.parse_args(argv)
    print(dump_config_value(args.key, args.config))


if __name__ == "__main__":
    main()
