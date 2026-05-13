# emoSVG 课题组服务器跑通指引

本文档面向 Linux GPU 服务器，目标是在课题组服务器上把 emoSVG 从“能 import / 能 fallback 跑通”逐步推进到“能调用真实模型生成动画、3D 和 SVG”。

建议不要一上来直接跑完整 `/generate`。本项目依赖 LivePortrait、ToonCrafter、TripoSR、SAM 等多个外部模型，最稳的顺序是：

1. 先跑通 Python 环境和 CPU fallback。
2. 再安装第三方源码依赖。
3. 再下载模型权重。
4. 再逐个验证 SAM、TripoSR、LivePortrait、ToonCrafter。
5. 最后启动 FastAPI 服务并用接口测试。

---

## A. 推荐：从污染环境重建 clean env

如果你已经执行过第三方项目的完整 requirements，例如：

```bash
pip install -r ~/workspace/third_party/LivePortrait/requirements.txt
pip install -r ~/workspace/third_party/ToonCrafter/requirements.txt
```

并且看到过这些现象：

```text
torch 2.3.0 被换成 torch 2.11.0
torchvision 0.18.0 被换成 torchvision 0.26.0
numpy 1.26.4 被换成 numpy 1.24.2
Pillow / imageio / protobuf / transformers 被降级或升级
pip is looking at multiple versions ...
```

建议不要继续修旧环境，直接新建一个干净环境。第三方 zip、源码目录、模型权重不用删，只重建 Python 环境即可。

### A.1 保留已有文件

这些可以保留：

```text
~/workspace/emoSVG
~/workspace/third_party/
~/workspace/third_party_zips/
/data3/zhengmuhan/emosvg_models/
/data3/zhengmuhan/emosvg_outputs/
```

不建议保留的是已经被污染的 conda env。

### A.2 新建 conda 环境

```bash
conda deactivate
conda create -n zmhEmoSVG_clean python=3.10 -y
conda activate zmhEmoSVG_clean

cd ~/workspace/emoSVG
python -m pip install --upgrade pip setuptools wheel
```

### A.3 安装 CUDA 版 PyTorch

如果服务器 CUDA 12.1/12.x 可用，优先：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

如果是 CUDA 11.8：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
```

验证：

```bash
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

期望类似：

```text
torch: 2.3.0+cu121
torchvision: 0.18.0+cu121
cuda available: True
```

### A.4 安装 emoSVG 主项目依赖

CUDA 12.1：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

CUDA 11.8：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

然后把 NumPy 和 OpenCV 拉回项目推荐组合：

```bash
pip install "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall opencv-contrib-python opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证核心版本：

```bash
python - <<'PY'
import torch, torchvision
import numpy as np
import cv2
import PIL
import imageio

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)
print("PIL:", PIL.__version__)
print("imageio:", imageio.__version__)
PY
```

期望：

```text
torch: 2.3.0+cu121
torchvision: 0.18.0+cu121
cuda: True 12.1
numpy: 1.26.4
cv2: 4.9.0
```

### A.5 跑 fallback smoke test

先配置 `.env`，让 smoke test 能找到本地模型目录：

```bash
cp .env.example .env
```

至少确认：

```bash
MODELS_ROOT=/data3/zhengmuhan/workspace/emosvg_models
TRIPOSR_MODEL_PATH=/data3/zhengmuhan/workspace/emosvg_models/triposr
OUTPUT_ROOT=/data3/zhengmuhan/workspace/emosvg_outputs
```

新版 `scripts/smoke_test.py` 会自动读取项目根目录 `.env`。如果 `TRIPOSR_MODEL_PATH` 或 `MODELS_ROOT/triposr` 里存在 `config.yaml` 和 `model.ckpt`，TripoSR 会优先使用本地权重，不再去 HuggingFace Hub 查找。

然后跑项目自带 smoke test：

```bash
python scripts/smoke_test.py
```

期望：

```text
Result: 5/5 passed
```

如果这里失败，优先修主环境，不要继续接 LivePortrait/ToonCrafter。

### A.6 第三方源码只挂路径，不全量安装 requirements

把第三方源码目录挂到当前 Python 环境：

```bash
SITE_PACKAGES=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

echo "$HOME/workspace/third_party/TripoSR" > "$SITE_PACKAGES/emosvg_triposr.pth"
echo "$HOME/workspace/third_party/ToonCrafter" > "$SITE_PACKAGES/emosvg_tooncrafter.pth"
```

如果你的解压目录带 `-main`，就改成：

```bash
echo "$HOME/workspace/third_party/TripoSR-main" > "$SITE_PACKAGES/emosvg_triposr.pth"
echo "$HOME/workspace/third_party/ToonCrafter-main" > "$SITE_PACKAGES/emosvg_tooncrafter.pth"
```

LivePortrait 官方 zip 常见结构是：

```text
LivePortrait/src/live_portrait_pipeline.py
LivePortrait/src/config/inference_config.py
```

它的源码内部使用相对导入，所以不要直接把 `LivePortrait/src` 当成顶层模块导入。推荐创建一个兼容包壳：

```bash
mkdir -p ~/workspace/third_party/liveportrait_compat
cp -a ~/workspace/third_party/LivePortrait/src \
  ~/workspace/third_party/liveportrait_compat/liveportrait
touch ~/workspace/third_party/liveportrait_compat/liveportrait/__init__.py

echo "$HOME/workspace/third_party/liveportrait_compat" > "$SITE_PACKAGES/emosvg_liveportrait.pth"
```

如果目录名是 `LivePortrait-main`：

```bash
mkdir -p ~/workspace/third_party/liveportrait_compat
cp -a ~/workspace/third_party/LivePortrait-main/src \
  ~/workspace/third_party/liveportrait_compat/liveportrait
touch ~/workspace/third_party/liveportrait_compat/liveportrait/__init__.py

echo "$HOME/workspace/third_party/liveportrait_compat" > "$SITE_PACKAGES/emosvg_liveportrait.pth"
```

`segment-anything` 通常可以直接安装源码包：

```bash
pip install ~/workspace/third_party/segment-anything
```

如果目录是 `segment-anything-main`：

```bash
pip install ~/workspace/third_party/segment-anything-main
```

### A.7 验证第三方 import

```bash
python - <<'PY'
checks = []

try:
    import segment_anything
    checks.append(("segment_anything", "OK"))
except Exception as e:
    checks.append(("segment_anything", repr(e)))

try:
    import tsr
    checks.append(("TripoSR tsr", "OK"))
except Exception as e:
    checks.append(("TripoSR tsr", repr(e)))

try:
    from liveportrait.live_portrait_pipeline import LivePortraitPipeline
    checks.append(("LivePortrait", "OK"))
except Exception as e:
    checks.append(("LivePortrait", repr(e)))

try:
    import lvdm
    checks.append(("ToonCrafter lvdm", "OK"))
except Exception as e:
    checks.append(("ToonCrafter lvdm", repr(e)))

for name, result in checks:
    print(f"{name}: {result}")
PY
```

如果某个 import 缺少小包，优先单独安装缺失包，例如：

```bash
pip install tyro pykalman lmdb ffmpeg-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install decord omegaconf pytorch-lightning open_clip_torch timm av moviepy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

但不要安装会破坏主环境的包。尤其不要让第三方 requirements 改这些核心包：

```text
torch
torchvision
torchaudio
numpy
opencv-python
opencv-contrib-python
opencv-python-headless
transformers
tokenizers
Pillow
imageio
protobuf
setuptools
xformers
gradio
```

### A.8 如果确实要过滤安装第三方 requirements

只在 import 缺包较多时使用过滤安装。示例：

```bash
grep -v -E "^(torch|torchvision|torchaudio|numpy|opencv|opencv-python|opencv-contrib-python|opencv-python-headless|transformers|tokenizers|Pillow|imageio|protobuf|setuptools|xformers|gradio)" \
  ~/workspace/third_party/ToonCrafter/requirements.txt \
  > /tmp/tooncrafter_req_filtered.txt

pip install -r /tmp/tooncrafter_req_filtered.txt
```

安装后必须立刻复查核心版本：

```bash
python - <<'PY'
import torch, torchvision
import numpy as np
import cv2
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)
PY
```

如果发现 `torch`、`torchvision`、`numpy`、`cv2` 被改了，立即恢复：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall opencv-contrib-python opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### A.9 推荐长期方案：ToonCrafter 单独环境

ToonCrafter 的依赖最容易污染主环境。如果后续要真实启用 ToonCrafter，推荐单独建环境：

```text
zmhEmoSVG_clean      # emoSVG 主服务：FastAPI / SAM / TripoSR / LivePortrait
zmhToonCrafter       # ToonCrafter 官方环境
```

主服务通过命令行、HTTP 或中间文件调用 ToonCrafter，避免它的 `torch/numpy/transformers/xformers` 版本污染 emoSVG 主环境。

---

## 0. 推荐服务器配置

| 项目 | 建议 |
|---|---|
| Python | 3.10 优先，3.11 也可尝试 |
| GPU | NVIDIA GPU，建议 10 GB VRAM 以上 |
| CUDA | 11.8 或 12.1/12.x |
| RAM | 16 GB 起，32 GB 更稳 |
| 磁盘 | 至少 40 GB 可用空间，模型和环境会占较多空间 |
| 进程数 | API 服务必须 `workers=1` |

如果服务器是 Slurm 集群，先申请交互式 GPU 节点，例如：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
```

具体 partition 名称按课题组服务器实际配置修改。

---

## 1. 进入项目目录

假设项目放在：

```bash
/data3/你的用户名/workspace/emoSVG
```

进入项目根目录：

```bash
cd /data3/你的用户名/workspace/emoSVG
pwd
ls
```

项目根目录应能看到：

```text
README.md
requirements.txt
pyproject.toml
src/
tests/
scripts/
configs/
docs/
models/
```

如果服务器 git 下载很慢，可以在本地网页下载 zip，再 `scp` 到服务器解压：

```bash
scp emoSVG.zip user@server:/data3/你的用户名/workspace/
unzip emoSVG.zip
cd emoSVG
```

---

## 2. 检查 GPU 和 CUDA

```bash
nvidia-smi
```

记录 CUDA 版本和显存大小。之后安装 PyTorch 时按 CUDA 版本选 wheel。

检查 Python：

```bash
python --version
which python
```

如果服务器默认 Python 版本不合适，建议用 conda 创建环境。

---

## 3. 创建隔离环境

推荐 conda：

```bash
conda create -n emosvg python=3.10 -y
conda activate emosvg
python -m pip install --upgrade pip setuptools wheel
```

如果服务器没有 conda，也可以用 venv：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

确认当前环境：

```bash
which python
python --version
pip --version
```

---

## 4. 安装 PyTorch

根据 `nvidia-smi` 显示的 CUDA 情况选择一个命令。

CUDA 12.1 常用：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8 常用：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
```

CUDA 13.0 使用

````
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
````

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
```

如果 `torch.cuda.is_available()` 是 `False`，先不要继续下载大模型，优先解决 PyTorch/CUDA 环境。

---

## 5. 安装项目基础依赖

项目依赖在 `requirements.txt` 中。由于 PyTorch 已经单独安装，下面命令会复用已安装的 CUDA 版 torch：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

如果你的服务器是 CUDA 11.8，把上面的 `cu121` 改成 `cu118`：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

13.0

 ````
 pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
 pip install -e .
 ````

建议显式钉住 NumPy 与 OpenCV 版本，避免服务器环境里混入较新的 `opencv-contrib-python` 后触发 NumPy 2.x 兼容问题：

```bash
pip install "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall opencv-contrib-python opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证版本：

```bash
python - <<'PY'
import numpy as np
import cv2
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)
PY
```

期望看到：

```text
numpy: 1.26.4
cv2: 4.9.0
```

基础 import 验证：

```bash
python - <<'PY'
import cv2
import fastapi
import numpy
import PIL
import torch
import trimesh
print("base deps OK")
print("torch cuda:", torch.cuda.is_available())
PY
```

---

## 6. 先跑 CPU fallback smoke test

这一步不需要真实模型权重，目的是确认 repo 本身能跑。

```bash
python scripts/smoke_test.py
```

期望结果类似：

```text
Result: 5/5 passed
```

也可以跑测试：

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
```

如果这一步失败，先看 import error 或基础依赖问题，不要急着下载模型。

---

## 7. 安装第三方源码依赖

本项目会用到几个上游项目：

- `segment-anything`
- `TripoSR`
- `LivePortrait`
- `ToonCrafter`

### 7.1 优先尝试脚本安装

如果服务器能访问 GitHub：

```bash
GITHUB_SSH=0 python scripts/download_models.py --models git_packages
```

如果 GitHub 很慢，可以设置镜像：

```bash
GITHUB_SSH=0 GITHUB_MIRROR=https://github.com python scripts/download_models.py --models git_packages
```

安装后验证：

```bash
python - <<'PY'
checks = []

try:
    import segment_anything
    checks.append(("segment_anything", "OK"))
except Exception as e:
    checks.append(("segment_anything", repr(e)))

try:
    import tsr
    checks.append(("TripoSR tsr", "OK"))
except Exception as e:
    checks.append(("TripoSR tsr", repr(e)))

try:
    from liveportrait.live_portrait_pipeline import LivePortraitPipeline
    checks.append(("LivePortrait", "OK"))
except Exception as e:
    checks.append(("LivePortrait", repr(e)))

try:
    import lvdm
    checks.append(("ToonCrafter lvdm", "OK"))
except Exception as e:
    checks.append(("ToonCrafter lvdm", repr(e)))

for name, result in checks:
    print(f"{name}: {result}")
PY
```

### 7.2 如果脚本安装失败：手动 zip 安装

如果服务器 clone 速度很慢，建议在本地浏览器下载四个 zip，然后传到服务器：

- <https://github.com/facebookresearch/segment-anything>
- <https://github.com/VAST-AI-Research/TripoSR>
- <https://github.com/KwaiVGI/LivePortrait>
- <https://github.com/ToonCrafter/ToonCrafter>

服务器上建议放到持久目录，不要放 `/tmp`：

```bash
mkdir -p ~/workspace/third_party
```

解压后目录建议为：

```text
~/workspace/third_party/
  segment-anything/
  TripoSR/
  LivePortrait/
  ToonCrafter/
```

安装可直接 pip install 的包：

```bash
pip install ~/workspace/third_party/segment-anything
```

不要直接全量安装 TripoSR、LivePortrait、ToonCrafter 的 requirements；这些研究项目常会改掉 `torch`、`numpy`、`opencv`、`transformers` 等主环境核心依赖。优先只挂源码路径，缺什么小包再单独补。

把源码路径写入当前 Python 环境的 `site-packages`：

```bash
SITE_PACKAGES=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

echo "$HOME/workspace/third_party/TripoSR" > "$SITE_PACKAGES/emosvg_triposr.pth"
echo "$HOME/workspace/third_party/ToonCrafter" > "$SITE_PACKAGES/emosvg_tooncrafter.pth"
```

LivePortrait 官方源码需要兼容包壳：

```bash
mkdir -p ~/workspace/third_party/liveportrait_compat
cp -a ~/workspace/third_party/LivePortrait/src \
  ~/workspace/third_party/liveportrait_compat/liveportrait
touch ~/workspace/third_party/liveportrait_compat/liveportrait/__init__.py

echo "$HOME/workspace/third_party/liveportrait_compat" > "$SITE_PACKAGES/emosvg_liveportrait.pth"
pip install tyro pykalman lmdb ffmpeg-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

再次验证 import：

```bash
python - <<'PY'
import segment_anything
print("segment_anything OK")

import tsr
print("TripoSR OK")

from liveportrait.live_portrait_pipeline import LivePortraitPipeline
print("LivePortrait OK")

import lvdm
print("ToonCrafter lvdm OK")
PY
```

注意：当前 repo 的 `ToonCrafterWrapper` 代码尝试导入 `from tooncrafter.inference import ToonCrafterInference`。而 ToonCrafter 官方源码常见可用入口是 `lvdm`。因此如果 `import lvdm` 成功但真实 ToonCrafter wrapper 仍失败，这不是环境完全失败，而是说明 ToonCrafter adapter 还需要按当前上游 API 做一次适配。首轮跑通建议先关闭 ToonCrafter。

---

## 8. 配置模型和输出路径

建议把模型和输出放到有足够空间的持久目录：

```bash
mkdir -p /data3/你的用户名/emosvg_models
mkdir -p /data3/你的用户名/emosvg_outputs
```

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
DEVICE=cuda
TORCH_DTYPE=float16
VRAM_BUDGET_GB=10.0

MODELS_ROOT=/data3/你的用户名/emosvg_models
OUTPUT_ROOT=/data3/你的用户名/emosvg_outputs

API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_LEVEL=INFO
```

如果 HuggingFace 在服务器访问慢，可以增加：

```bash
HF_HOME=/data3/你的用户名/emosvg_models/.hf_cache
HF_ENDPOINT=https://hf-mirror.com
```

如果服务器需要 HuggingFace token：

```bash
huggingface-cli login
```

或在 `.env` 中设置：

```bash
HF_TOKEN=hf_xxxx
```

---

## 9. 下载模型权重

先加载 `.env`：

```bash
set -a
source .env
set +a
```

建议不要一开始下载全部，按模块逐个下载和验证。

### 9.1 下载 SAM

```bash
python scripts/download_models.py --models sam
```

期望文件：

```text
$MODELS_ROOT/sam/sam_vit_h_4b8939.pth
```

### 9.2 下载 TripoSR

```bash
python scripts/download_models.py --models triposr
```

期望文件：

```text
$MODELS_ROOT/triposr/config.yaml
$MODELS_ROOT/triposr/model.ckpt
```

### 9.3 下载 LivePortrait

```bash
python scripts/download_models.py --models live_portrait
```

期望文件：

```text
$MODELS_ROOT/live_portrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth
$MODELS_ROOT/live_portrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth
$MODELS_ROOT/live_portrait/pretrained_weights/liveportrait/base_models/warping_module.pth
$MODELS_ROOT/live_portrait/pretrained_weights/liveportrait/base_models/spade_generator.pth
$MODELS_ROOT/live_portrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth
```

### 9.4 下载 IP-Adapter

```bash
python scripts/download_models.py --models ip_adapter
```

期望目录：

```text
$MODELS_ROOT/ip_adapter/models/image_encoder/
$MODELS_ROOT/ip_adapter/ip-adapter-faceid_sd15.bin
```

### 9.5 下载 ToonCrafter

ToonCrafter 权重较大，建议最后下载：

```bash
python scripts/download_models.py --models toon_crafter
```

期望文件：

```text
$MODELS_ROOT/toon_crafter/model.ckpt
$MODELS_ROOT/toon_crafter/config.yaml
```

### 9.6 检查模型目录

```bash
find "$MODELS_ROOT" -maxdepth 4 -type f | sort | head -100
du -sh "$MODELS_ROOT"
```

注意：`scripts/download_models.py` 中部分下载函数会记录 error 后继续执行，所以下载后一定要人工检查文件是否真的存在。

---

## 10. 生成一张测试图

如果没有现成角色图，可以先生成一个简单 PNG：

```bash
mkdir -p /data3/你的用户名/emosvg_test

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

out = Path("/data3/你的用户名/emosvg_test/test_character.png")
out.parent.mkdir(parents=True, exist_ok=True)

img = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
d = ImageDraw.Draw(img)
d.ellipse((96, 80, 416, 400), fill=(255, 210, 80, 255), outline=(30, 30, 30, 255), width=8)
d.ellipse((180, 190, 225, 245), fill=(30, 30, 30, 255))
d.ellipse((287, 190, 332, 245), fill=(30, 30, 30, 255))
d.arc((185, 220, 330, 330), 10, 170, fill=(30, 30, 30, 255), width=8)
d.rectangle((210, 370, 302, 430), fill=(80, 160, 255, 255), outline=(30, 30, 30, 255), width=6)
img.save(out)
print(out)
PY
```

如果是测试真实效果，最好准备一张主体居中、背景简单、分辨率 512 或 1024 的 PNG。

---

## 11. 单模块验证顺序

### 11.1 先验证 API 能启动

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

另开一个终端：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/expressions
curl http://127.0.0.1:8000/status
```

期望 `/health` 返回：

```json
{"status":"ok"}
```

### 11.2 只跑动画，不跑 3D/SVG/ToonCrafter

这是第一条真实业务链路：

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "file=@/data3/你的用户名/emosvg_test/test_character.png" \
  -F "expression=shock" \
  -F "output_format=gif" \
  -F "run_3d=false" \
  -F "run_svg=false" \
  -F "use_toon_crafter=false" \
  -F "fps=12" \
  -F "width=512" \
  -F "height=512"
```

如果 LivePortrait 权重或包不可用，这一步仍可能走 fallback 并输出结构正确的动画文件。

### 11.3 只跑 SVG

```bash
curl -X POST http://127.0.0.1:8000/vectorize \
  -F "file=@/data3/你的用户名/emosvg_test/test_character.png" \
  -F "bezier_tolerance=2.0" \
  -F "min_region_area=100"
```

如果 SAM 权重不可用，会走 OpenCV fallback。

### 11.4 只跑 3D

```bash
curl -X POST http://127.0.0.1:8000/reconstruct \
  -F "file=@/data3/你的用户名/emosvg_test/test_character.png" \
  -F "mc_resolution=128" \
  -F "remove_background=true" \
  -F "foreground_ratio=0.85"
```

首次测试建议 `mc_resolution=128`，确认稳定后再提高到 256。

### 11.5 跑动画 + SVG

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "file=@/data3/你的用户名/emosvg_test/test_character.png" \
  -F "expression=laugh" \
  -F "output_format=gif" \
  -F "run_3d=false" \
  -F "run_svg=true" \
  -F "use_toon_crafter=false" \
  -F "fps=12" \
  -F "width=512" \
  -F "height=512"
```

### 11.6 最后再尝试 ToonCrafter

ToonCrafter 权重大、耗显存，且当前 wrapper 与上游 API 可能需要适配。建议最后测试：

```bash
curl -X POST http://127.0.0.1:8000/animate \
  -F "file=@/data3/你的用户名/emosvg_test/test_character.png" \
  -F "expression=shock" \
  -F "output_format=gif" \
  -F "use_toon_crafter=true" \
  -F "frames_between=2" \
  -F "fps=12" \
  -F "width=512" \
  -F "height=512"
```

如果报 `No module named tooncrafter` 或 `ToonCrafterInference` 不存在，但 `import lvdm` 成功，说明主要问题在 adapter 对接，而不是权重或基础环境。

---

## 12. 长时间运行服务

推荐用 `tmux`：

```bash
tmux new -s emosvg
conda activate emosvg
cd /data3/你的用户名/workspace/emoSVG
set -a
source .env
set +a
mkdir -p logs
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 2>&1 | tee logs/server.log
```

退出 tmux 但保持服务运行：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
tmux attach -t emosvg
```

如果希望在本地浏览器打开服务器上的 Swagger UI，可以做 SSH 端口转发：

```bash
ssh -L 8000:127.0.0.1:8000 user@server
```

然后本地浏览器打开：

```text
http://127.0.0.1:8000/docs
```

---

## 13. 显存策略

当前各模型大致显存预算：

| 模型 | 估计 VRAM |
|---|---:|
| IPExtractor / CLIP | 2.5 GB |
| LivePortrait | 4.5 GB |
| ToonCrafter | 8.0 GB |
| TripoSR | 6.0 GB |
| SAM ViT-H | 代码中按 6.5 GB 预算 |

推荐 `.env`：

```bash
VRAM_BUDGET_GB=10.0
TORCH_DTYPE=float16
API_WORKERS=1
```

如果 GPU 只有 8 GB：

```bash
VRAM_BUDGET_GB=7.0
```

并优先关闭 ToonCrafter 和 3D：

```bash
run_3d=false
use_toon_crafter=false
```

如果 GPU 只有 6 GB 左右，建议先只跑 fallback、LivePortrait 或 SVG fallback，不要直接跑 ToonCrafter。

---

## 14. 常见问题

### 14.1 `torch.cuda.is_available()` 是 False

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

常见原因：

- 安装了 CPU 版 torch。
- 当前节点没有分配 GPU。
- CUDA driver 太旧。
- conda 环境不是当前 shell 使用的环境。

### 14.2 `No module named segment_anything`

重新安装：

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

或用前文 zip + `pip install ~/workspace/third_party/segment-anything`。

### 14.3 `No module named tsr`

说明 TripoSR 源码没有进入 Python path。检查 `.pth`：

```bash
python - <<'PY'
import site, pathlib
sp = pathlib.Path(site.getsitepackages()[0])
print(sp)
print(list(sp.glob("emosvg_*.pth")))
PY
```

确认 `emosvg_triposr.pth` 指向持久存在的 TripoSR 源码目录。

### 14.4 `No module named liveportrait`

LivePortrait 官方 zip 常见结构是 `LivePortrait/src/live_portrait_pipeline.py`，内部使用相对导入。不要直接把 `.pth` 指向 `LivePortrait/src`，推荐确认 `.pth` 指向兼容包壳的上一级目录：

```bash
cat $(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)/emosvg_liveportrait.pth
```

期望类似：

```text
/data3/你的用户名/workspace/third_party/liveportrait_compat
```

如果还没有创建兼容包壳：

```bash
mkdir -p ~/workspace/third_party/liveportrait_compat
cp -a ~/workspace/third_party/LivePortrait/src \
  ~/workspace/third_party/liveportrait_compat/liveportrait
touch ~/workspace/third_party/liveportrait_compat/liveportrait/__init__.py
```

如果随后报 `No module named 'tyro'`，补小依赖：

```bash
pip install tyro pykalman lmdb ffmpeg-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 14.5 ToonCrafter 可以 `import lvdm`，但 emoSVG 仍报错

当前 `src/modules/meme_animator/toon_crafter.py` 期待：

```python
from tooncrafter.inference import ToonCrafterInference
```

但 ToonCrafter 官方源码常见入口是 `lvdm`，不一定有 `tooncrafter.inference` 这个标准包结构。此时建议：

1. 首轮跑通时设置 `use_toon_crafter=false`。
2. 先确认 LivePortrait、TripoSR、SAM 都能跑。
3. 再单独为 ToonCrafter 写 adapter，把官方 inference 脚本封装成 `generate(frame0, frame1, num_frames)` 或 `interpolate(frame0, frame1, num_frames)`。

### 14.6 HuggingFace 下载很慢或失败

可尝试：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data3/你的用户名/emosvg_models/.hf_cache
```

也可以手动：

```bash
huggingface-cli download stabilityai/TripoSR --local-dir $MODELS_ROOT/triposr
huggingface-cli download KwaiVGI/LivePortrait --local-dir $MODELS_ROOT/live_portrait
huggingface-cli download Doubiiu/ToonCrafter --local-dir $MODELS_ROOT/toon_crafter
```

### 14.7 CUDA out of memory

处理顺序：

1. 确认 `uvicorn` 是 `--workers 1`。
2. 降低 `VRAM_BUDGET_GB`。
3. 降低 `width` / `height` 到 512 或 256。
4. 先关 `use_toon_crafter`。
5. 再关 `run_3d`。
6. 用 `nvidia-smi` 查是否有别人的进程占显存。

### 14.8 端口被占用

换端口：

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8010 --workers 1
```

本地转发也对应改：

```bash
ssh -L 8010:127.0.0.1:8010 user@server
```

### 14.9 `/tmp` 目录重启后丢失

不要把第三方源码或模型放在 `/tmp`。建议：

```text
/data3/你的用户名/workspace/third_party/
/data3/你的用户名/emosvg_models/
/data3/你的用户名/emosvg_outputs/
```

### 14.10 NumPy 2.x 与 OpenCV/3D 依赖兼容问题

如果 `python scripts/smoke_test.py` 在 `Full pipeline (3D+SVG)` 阶段报错：

```text
`ptp` was removed from the ndarray class in NumPy 2.0.
```

说明当前环境里有依赖尚未适配 NumPy 2.x。对本项目这类视觉、3D、几何 pipeline，更推荐使用 NumPy 1.26 系列：

```bash
pip install "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果降级 NumPy 时看到类似提示：

```text
opencv-contrib-python 4.13.0.92 requires numpy>=2
```

说明环境里混入了不属于本项目 requirements 的新版 OpenCV contrib 包。建议清理 OpenCV 混装，并安装项目指定版本：

```bash
pip uninstall opencv-contrib-python opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```bash
python - <<'PY'
import numpy as np
import cv2
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)
PY
```

期望：

```text
numpy: 1.26.4
cv2: 4.9.0
```

### 14.11 TripoSR `has_vertex_color` 签名兼容问题

如果 smoke test 或 `/reconstruct` 报错：

```text
TripoSR inference failed: TSR.extract_mesh() missing 1 required positional argument: 'has_vertex_color'
```

说明当前安装的 TripoSR 上游版本要求：

```python
extract_mesh(..., has_vertex_color=False)
```

而旧版 emoSVG wrapper 只传了 `resolution`。新版 wrapper 已经兼容新旧两种签名。遇到该错误时，先更新：

```text
src/modules/reconstructor_3d/reconstructor.py
```

然后重跑：

```bash
python scripts/smoke_test.py
```

这不是权重下载问题，也不是环境依赖问题。

如果报错：

```text
Failed to load TripoSR from 'stabilityai/TripoSR': An error happened while trying to locate the file on the Hub
```

说明程序仍在使用 HuggingFace ID，而没有找到本地权重。检查：

```bash
ls -lh /data3/zhengmuhan/workspace/emosvg_models/triposr/config.yaml
ls -lh /data3/zhengmuhan/workspace/emosvg_models/triposr/model.ckpt
cat .env | grep -E "MODELS_ROOT|TRIPOSR_MODEL_PATH"
```

确保 `.env` 包含：

```bash
MODELS_ROOT=/data3/zhengmuhan/workspace/emosvg_models
TRIPOSR_MODEL_PATH=/data3/zhengmuhan/workspace/emosvg_models/triposr
```

---

### 14.12 TripoSR `expected scalar type Half but found Float`

如果 smoke test 已经能加载本地 TripoSR 权重，但报错：

```text
TripoSR inference failed: expected scalar type Half but found Float
```

这说明 TripoSR 已进入真实推理阶段，问题不再是下载或路径，而是 dtype 不一致：全局配置里的 `TORCH_DTYPE=float16` 会把模型移到 half，但 TripoSR 上游预处理路径仍会生成 float32 输入。新版 emoSVG wrapper 默认会让 TripoSR 保持 float32，这在 24 GB 4090 上更稳。

先同步新版：

```text
src/modules/reconstructor_3d/reconstructor.py
```

然后在 `.env` 中可以显式加入：

```bash
TRIPOSR_FORCE_FLOAT32=1
```

`1` 是新版默认值。除非你确认当前 TripoSR 源码和输入预处理都支持 half 推理，否则不要设成 `0`。

再次运行：

```bash
python scripts/smoke_test.py
```

---

## 15. 推荐验收清单

按顺序勾选：

- [ ] `nvidia-smi` 正常。
- [ ] `torch.cuda.is_available()` 为 `True`。
- [ ] `python scripts/smoke_test.py` 通过。
- [ ] `pip install -e .` 完成。
- [ ] `import segment_anything` 成功。
- [ ] `import tsr` 成功。
- [ ] `from liveportrait.live_portrait_pipeline import LivePortraitPipeline` 成功。
- [ ] `import lvdm` 成功，或明确暂不启用 ToonCrafter。
- [ ] `$MODELS_ROOT/sam/sam_vit_h_4b8939.pth` 存在。
- [ ] `$MODELS_ROOT/triposr/model.ckpt` 存在。
- [ ] LivePortrait 五个 `.pth` 权重存在。
- [ ] `curl /health` 返回 ok。
- [ ] `/generate` 在 `run_3d=false run_svg=false use_toon_crafter=false` 下返回动画。
- [ ] `/vectorize` 能返回 SVG。
- [ ] `/reconstruct` 能返回 OBJ/GLB。
- [ ] ToonCrafter 如需启用，已完成 adapter 适配或确认 wrapper 可用。

---

## 16. 建议首轮跑通目标

首轮不建议追求“全部模型同时真实推理”。更稳的阶段目标是：

1. **第 1 阶段**：无权重 fallback 全部通过。
2. **第 2 阶段**：SAM 真实分割 + SVG 输出。
3. **第 3 阶段**：TripoSR 真实 3D 输出。
4. **第 4 阶段**：LivePortrait 真实动画输出。
5. **第 5 阶段**：ToonCrafter adapter 修通后再接入补帧/首末帧生成。

这样每一步失败时都能明确定位到环境、权重、上游包、adapter 或显存问题。
