"""Config merge: machine ← profile ← experiment ← config.local.toml."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# No machine name is hardcoded here: a fresh clone defaults to the generic,
# hardware-agnostic `configs/machine/example.toml` profile. Set this env var
# (e.g. in your shell rc, not in the repo) to make your own machine profile
# the default without touching any committed file or CLI invocation.
DEFAULT_MACHINE_ENV = "ES_CAPACITY_MACHINE"
DEFAULT_MACHINE = "example"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class ModelSpec:
    key: str
    path: Path
    hf_id: str = ""

    @property
    def resolved(self) -> Path:
        return self.path


@dataclass
class AppConfig:
    raw: dict[str, Any]
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)

    @property
    def hash(self) -> str:
        return config_hash(self.raw)

    @property
    def models_dir(self) -> Path:
        p = Path(self.raw.get("paths", {}).get("models_dir", "models"))
        return p if p.is_absolute() else self.repo_root / p

    @property
    def runs_dir(self) -> Path:
        p = Path(self.raw.get("paths", {}).get("runs_dir", "runs"))
        return p if p.is_absolute() else self.repo_root / p

    @property
    def data_dir(self) -> Path:
        p = Path(self.raw.get("paths", {}).get("data_dir", "data"))
        return p if p.is_absolute() else self.repo_root / p

    @property
    def venv_dir(self) -> Path | None:
        """Optional `paths.venv` from config.local.toml; None if unset."""
        raw = self.raw.get("paths", {}).get("venv")
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else self.repo_root / p

    def model(self, key: str) -> ModelSpec:
        models = self.raw.get("models", {})
        if key not in models:
            raise KeyError(f"Unknown model key {key!r}; known={list(models)}")
        m = models[key]
        rel = m.get("path", key)
        path = Path(rel)
        if not path.is_absolute():
            path = self.models_dir / path
        return ModelSpec(key=key, path=path, hf_id=m.get("hf_id", ""))

    def section(self, *keys: str) -> dict[str, Any]:
        cur: Any = self.raw
        for k in keys:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return dict(cur) if isinstance(cur, dict) else {}


def load_config(
    *,
    machine: str | None = None,
    profile: str | None = None,
    experiment: str | None = None,
    local_path: Path | None = None,
) -> AppConfig:
    root = REPO_ROOT
    machine = machine or os.environ.get(DEFAULT_MACHINE_ENV) or DEFAULT_MACHINE
    cfg: dict[str, Any] = {}
    cfg = _deep_merge(cfg, _load_toml(root / "configs" / "machine" / f"{machine}.toml"))
    if profile:
        cfg = _deep_merge(cfg, _load_toml(root / "configs" / "profiles" / f"{profile}.toml"))
    if experiment:
        cfg = _deep_merge(cfg, _load_toml(root / "configs" / "experiments" / f"{experiment}.toml"))
    local = local_path or (root / "config.local.toml")
    cfg = _deep_merge(cfg, _load_toml(local))
    return AppConfig(raw=cfg, repo_root=root)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Dump config values")
    p.add_argument(
        "--machine",
        default=None,
        help=f"Defaults to ${DEFAULT_MACHINE_ENV} env var, else {DEFAULT_MACHINE!r}",
    )
    p.add_argument("--profile", default=None)
    p.add_argument("--experiment", default=None)
    p.add_argument(
        "--model-key",
        default=None,
        help="Print the resolved absolute path for models.<key> and exit (for use in shell scripts)",
    )
    p.add_argument("key", nargs="?", default=None, help="dot.path e.g. paths.models_dir")
    args = p.parse_args(argv)
    cfg = load_config(machine=args.machine, profile=args.profile, experiment=args.experiment)
    if args.model_key is not None:
        print(cfg.model(args.model_key).resolved)
        return
    if args.key is None:
        print(json.dumps(cfg.raw, indent=2, default=str))
        return
    cur: Any = cfg.raw
    for part in args.key.split("."):
        cur = cur[part]
    if isinstance(cur, (dict, list)):
        print(json.dumps(cur, indent=2, default=str))
    else:
        print(cur)


if __name__ == "__main__":
    main()
