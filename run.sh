#!/usr/bin/env bash
# 始终通过项目内的 venv 解释器运行 train_nextai.py，避免与系统/全局 pip 混淆。
# 用法： ./run.sh               -> 执行训练
#        ./run.sh --only-check -> 仅打印 torch + transformers + datasets 版本，快速自检
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "[run.sh] 未发现 .venv，正在为当前项目创建虚拟环境..."
  python3 -m venv "${PROJECT_ROOT}/.venv"
  "${VENV_PY}" -m pip install --upgrade pip
fi

# 如果关键依赖缺失，则自动安装（只在首次运行时发生）
if ! "${VENV_PY}" -c "import torch, transformers, datasets" 2>/dev/null; then
  echo "[run.sh] 关键依赖缺失，正在从 requirements.txt 安装..."
  "${VENV_PY}" -m pip install -r "${REQUIREMENTS}"
fi

if [[ "${1:-}" == "--only-check" ]]; then
  "${VENV_PY}" -c "import torch, transformers, datasets
print('python      :', __import__('sys').version.split()[0])
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('datasets    :', datasets.__version__)"
  exit 0
fi

cd "${PROJECT_ROOT}"
exec "${VENV_PY}" "${PROJECT_ROOT}/train_nextai.py" "$@"
