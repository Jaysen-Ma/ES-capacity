#!/usr/bin/env bash
# Shared environment bootstrap, sourced (not executed) by scripts/*.sh.
# Never hardcodes a machine-specific path so the repo works unmodified after
# `git clone`; personal paths stay in config.local.toml (gitignored).
#
# venv resolution order (first match wins):
#   1. Already-active venv ($VIRTUAL_ENV) -> no-op.
#   2. $ES_CAPACITY_VENV env var (set in your own shell rc, not in the repo).
#   3. `paths.venv` in config.local.toml (see config.local.example.toml).
#   4. `.venv/` in the repo root (`python3 -m venv .venv`).
#   5. Otherwise: warn and continue, assuming deps are already on PATH.
#
# Must be sourced with $0 (or ${BASH_SOURCE[0]}) still pointing at this file,
# e.g.: source "$SCRIPT_DIR/_env.sh"

ES_CAPACITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ES_CAPACITY_ROOT

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  venv_path="${ES_CAPACITY_VENV:-}"
  if [[ -z "$venv_path" ]]; then
    venv_path="$(PYTHONPATH="$ES_CAPACITY_ROOT" python3 -m es_capacity.config paths.venv 2>/dev/null || true)"
  fi
  if [[ -z "$venv_path" && -d "$ES_CAPACITY_ROOT/.venv" ]]; then
    venv_path="$ES_CAPACITY_ROOT/.venv"
  fi
  if [[ -n "$venv_path" && -f "$venv_path/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
  else
    echo "[env] no venv resolved; assuming dependencies are already on PATH" \
      "(set \$ES_CAPACITY_VENV or paths.venv in config.local.toml to activate one)" >&2
  fi
fi
