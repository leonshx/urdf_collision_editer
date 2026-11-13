#!/usr/bin/env bash
# One-command local deployment and launcher for URDF Collision Editor.
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "错误：未找到 uv，且系统中没有 python3，无法自动安装部署工具。" >&2
    exit 1
  fi
  if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "错误：未找到 uv，且 python3 未安装 pip。请先安装 uv：https://docs.astral.sh/uv/" >&2
    exit 1
  fi

  echo "未找到 uv，正在通过 pip 安装到当前用户目录…" >&2
  python3 -m pip install --user --upgrade uv >&2
  if command -v uv >/dev/null 2>&1; then
    command -v uv
  elif [[ -x "$HOME/.local/bin/uv" ]]; then
    printf '%s\n' "$HOME/.local/bin/uv"
  else
    echo "错误：uv 安装完成但无法找到可执行文件。" >&2
    exit 1
  fi
}

UV_BIN="$(find_uv)"
echo "正在同步 Python 环境与项目依赖…"
"$UV_BIN" sync --locked

if [[ "${1:-}" == "--setup-only" ]]; then
  echo "部署完成。运行 ./deploy.sh --urdf /path/to/robot.urdf 启动编辑器。"
  exit 0
fi

echo "正在启动 URDF 碰撞体编辑器…"
exec "$UV_BIN" run urdf-collision-editor "$@"
