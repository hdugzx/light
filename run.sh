#!/usr/bin/env bash
# Run the terminal agent. Uses a Python that already has openai + opencv + pillow
# (defaults to the .venv-vllm interpreter). Args are forwarded to `python -m light_agent`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${VENV_PY:-/mnt/orisd/shuaiman/.venv-vllm/bin/python}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "找不到 venv 解释器: $VENV_PY" >&2
  echo "可用 VENV_PY=/path/to/python 覆盖，或 pip install -r requirements.txt 后用系统 python。" >&2
  exit 1
fi

cd "$HERE"
exec "$VENV_PY" -m light_agent "$@"
