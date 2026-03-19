#!/usr/bin/env bash
set -euo pipefail

# 用 uv 管理 Python 环境（推荐在仓库根目录执行）

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv，请先安装 uv 后再运行本脚本。" >&2
  exit 1
fi

# 减少安装时的进度条输出（避免终端日志过长）
export UV_NO_PROGRESS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# 1) 创建/复用虚拟环境
# 优先使用 Python 3.10；如果宿主机未预装，则回退到 uv 默认 Python。
if uv venv -p 3.10 .venv; then
  :
else
  echo "[WARN] 宿主机未找到 Python 3.10，回退到 uv 默认 Python 创建 .venv" >&2
  uv venv .venv
fi
source .venv/bin/activate

# 2) Python 依赖
uv pip install --upgrade pip
uv pip install "numpy<2" tqdm plyfile pillow opencv-python matplotlib

# 3) PyTorch（CUDA 12.1）
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# 4) CUDA 扩展（本地子模块）
uv pip install "pip<24"
uv pip install wheel ninja

uv pip install setuptools==68.2.2
uv pip install -e submodules/diff-gaussian-rasterization/ --no-build-isolation
uv pip install -e submodules/simple-knn/ --no-build-isolation
uv pip install -e submodules/fused-ssim/ --no-build-isolation
uv pip install -e submodules/masked-diff-gaussian-rasterization/ --no-build-isolation

# 5) 运行时系统依赖（可选）
if command -v apt >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y libgl1-mesa-glx libglib2.0-0
fi

# 6) 示例
# 生成 SAM 多视图一致 mask：
# python generate_multiview_sam_masks.py --source_path data/gs_data/bicycle \
#   --sam_checkpoint /path/to/sam_vit_h_4b8939.pth --sam_model_type vit_h
#
# 仅用指定物体 mask 训练（重建单物体）：
# python train.py -s data/gs_data/bicycle -m output/bicycle_mask_only \
#   --mask_only --mask_object_id 0 --mask_use_masked_rasterizer
