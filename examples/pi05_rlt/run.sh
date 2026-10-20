#!/usr/bin/env bash
set -euo pipefail

rlt_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
rlt_python="${RLT_PYTHON:-/home/robot/miniconda3/envs/lerobot/bin/python}"
rlt_command="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$rlt_command" in
  precompute|train_stage1|train_stage2|encode_cache|train_critical|export_policy)
    rlt_entry=(-m "lerobot.scripts.rlt.$rlt_command") ;;
  smoke)
    rlt_entry=("$rlt_repo/examples/pi05_rlt/smoke_test.py") ;;
  *)
    echo "Usage: bash examples/pi05_rlt/run.sh {precompute|train_stage1|train_stage2|encode_cache|train_critical|export_policy|smoke} [arguments]" >&2
    exit 2 ;;
esac
export PYTHONPATH="$rlt_repo/src"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
exec uv run --no-project --python "$rlt_python" python "${rlt_entry[@]}" "$@"
