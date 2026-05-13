# emoSVG 新环境构建指引

本文档用于从一个已经被第三方 requirements 污染的环境，重新构建一个可控的 emoSVG clean env。

适用情况：

```text
torch / torchvision 被 ToonCrafter 或 LivePortrait requirements 改掉
numpy 被降到 1.24.x 或升到 2.x
opencv-python / opencv-contrib-python / opencv-python-headless 混装
pip 一直在 looking at multiple versions
smoke test 因 NumPy / OpenCV / 3D 依赖失败
```

核心原则：

1. 主环境固定 `torch / torchvision / numpy / opencv / transformers`。
2. 不直接安装第三方论文项目的完整 `requirements.txt`。
3. 第三方源码优先用 `.pth` 挂路径。
4. 缺什么小包补什么小包。
5. ToonCrafter 如冲突严重，单独环境服务化。

---

## 1. 保留已有文件

这些目录可以保留，不需要重新下载：

```text
~/workspace/emoSVG
~/workspace/third_party/
~/workspace/third_party_zips/
/data3/zhengmuhan/emosvg_models/
/data3/zhengmuhan/emosvg_outputs/
```

不建议继续修已经污染的 conda env，直接新建环境更快。

---

## 2. 新建 clean conda 环境

```bash
conda deactivate
conda create -n zmhEmoSVG_clean python=3.10 -y
conda activate zmhEmoSVG_clean

cd ~/workspace/emoSVG
python -m pip install --upgrade pip setuptools wheel
```

---

## 3. 安装 CUDA 版 PyTorch

CUDA 12.1/12.x：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8：

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

期望：

```text
torch: 2.3.0+cu121
torchvision: 0.18.0+cu121
cuda available: True
```

---

## 4. 安装 emoSVG 主依赖

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

固定 NumPy 与 OpenCV：

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

---

## 5. 先跑 fallback smoke test

建议先配置 `.env`，让 smoke test 能找到本地模型目录：

```bash
cp .env.example .env
```

至少确认：

```bash
MODELS_ROOT=/data3/zhengmuhan/workspace/emosvg_models
LIVE_PORTRAIT_MODEL_PATH=/data3/zhengmuhan/workspace/emosvg_models/live_portrait
TRIPOSR_MODEL_PATH=/data3/zhengmuhan/workspace/emosvg_models/triposr
TOON_CRAFTER_MODEL_PATH=/data3/zhengmuhan/workspace/emosvg_models/toon_crafter
SAM_MODEL_PATH=/data3/zhengmuhan/workspace/emosvg_models/sam/sam_vit_h_4b8939.pth
OUTPUT_ROOT=/data3/zhengmuhan/workspace/emosvg_outputs
```

新版 `scripts/smoke_test.py` 会自动读取项目根目录 `.env`。如果 `LIVE_PORTRAIT_MODEL_PATH`、`TOON_CRAFTER_MODEL_PATH`、`SAM_MODEL_PATH`、`TRIPOSR_MODEL_PATH` 指向真实权重，smoke test 会优先尝试真实后端；缺失的模块会继续走 fallback。

```bash
python scripts/smoke_test.py
```

期望：

```text
Result: 5/5 passed
```

如果这里失败，先修主环境，不要继续安装 LivePortrait / ToonCrafter 依赖。

---

## 6. 挂第三方源码路径

不要直接执行：

```bash
pip install -r ~/workspace/third_party/LivePortrait/requirements.txt
pip install -r ~/workspace/third_party/ToonCrafter/requirements.txt
```

改为把源码路径写入当前环境：

```bash
SITE_PACKAGES=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

echo "$HOME/workspace/third_party/TripoSR" > "$SITE_PACKAGES/emosvg_triposr.pth"
echo "$HOME/workspace/third_party/ToonCrafter" > "$SITE_PACKAGES/emosvg_tooncrafter.pth"
```

如果目录名是 `*-main`：

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

安装 `segment-anything`：

```bash
pip install ~/workspace/third_party/segment-anything
```

或：

```bash
pip install ~/workspace/third_party/segment-anything-main
```

---

## 7. 验证第三方 import

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

如果缺少小包，优先单独安装：

```bash
pip install tyro pykalman lmdb ffmpeg-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install decord omegaconf pytorch-lightning open_clip_torch timm av moviepy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装后马上复查核心版本：

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

---

## 8. 如果必须安装第三方 requirements

只用过滤安装，并过滤会破坏主环境的包：

```bash
grep -v -E "^(torch|torchvision|torchaudio|numpy|opencv|opencv-python|opencv-contrib-python|opencv-python-headless|transformers|tokenizers|Pillow|imageio|protobuf|setuptools|xformers|gradio)" \
  ~/workspace/third_party/ToonCrafter/requirements.txt \
  > /tmp/tooncrafter_req_filtered.txt

pip install -r /tmp/tooncrafter_req_filtered.txt
```

不要让第三方 requirements 改掉这些核心包：

```text
torch
torchvision
torchaudio
numpy
opencv
transformers
tokenizers
Pillow
imageio
protobuf
xformers
gradio
```

如果被改坏，恢复：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall opencv-contrib-python opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 8.5 ToonCrafter 权重下载

ToonCrafter 建议最后处理。新版 `scripts/download_models.py` 会从 HF 模型仓库下载大权重 `model.ckpt`，再从官方 Space/GitHub 补 `configs/inference_512_v1.0.yaml`，并复制成 emoSVG 期望的 `config.yaml`。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data3/zhengmuhan/workspace/emosvg_models/.hf_cache
python scripts/download_models.py --models toon_crafter
```

成功后至少应存在：

```bash
ls -lh "$TOON_CRAFTER_MODEL_PATH/model.ckpt"
ls -lh "$TOON_CRAFTER_MODEL_PATH/config.yaml"
```

如果脚本失败，手动分两步：

```bash
mkdir -p "$TOON_CRAFTER_MODEL_PATH"

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Doubiiu/ToonCrafter \
  model.ckpt \
  --repo-type model \
  --local-dir "$TOON_CRAFTER_MODEL_PATH" \
  --resume-download

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Doubiiu/tooncrafter \
  configs/inference_512_v1.0.yaml \
  --repo-type space \
  --local-dir "$TOON_CRAFTER_MODEL_PATH/_space" \
  --resume-download

cp "$TOON_CRAFTER_MODEL_PATH/_space/configs/inference_512_v1.0.yaml" \
   "$TOON_CRAFTER_MODEL_PATH/config.yaml"
```

注意：`import lvdm OK` 只说明官方源码路径接好了；真实启用 ToonCrafter 还需要 wrapper 和官方推理入口适配。

---

## 9. 推荐长期结构

```text
zmhEmoSVG_clean
  emoSVG 主服务
  FastAPI
  SAM
  TripoSR
  LivePortrait

zmhToonCrafter
  ToonCrafter 官方环境
  独立脚本或服务
```

ToonCrafter 依赖最容易污染主环境。建议后续把它单独服务化，由 emoSVG 主服务通过命令行、HTTP 或中间文件调用。

---

## 10. TripoSR API 签名兼容问题

如果 smoke test 或 `/reconstruct` 报错：

```text
TripoSR inference failed: TSR.extract_mesh() missing 1 required positional argument: 'has_vertex_color'
```

说明当前安装的 TripoSR 上游版本要求：

```python
extract_mesh(..., has_vertex_color=False)
```

而旧代码只传了 `resolution`。新版 emoSVG wrapper 已经兼容这两种签名。遇到该错误时，先更新本项目代码中的：

```text
src/modules/reconstructor_3d/reconstructor.py
```

然后重跑：

```bash
python scripts/smoke_test.py
```

这不是权重下载问题，也不是环境依赖问题。

如果开头显示：

```text
LivePortrait weights not found at 'None'
ToonCrafter weights not found at 'None'
SAM weights not found
```

说明 smoke test 没有拿到这些模块的权重路径。请同步新版 `scripts/smoke_test.py` 和 `configs/base.yaml`，并检查 `.env` 里的 `LIVE_PORTRAIT_MODEL_PATH`、`TOON_CRAFTER_MODEL_PATH`、`SAM_MODEL_PATH`。

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

## 11. TripoSR Half/Float dtype 兼容问题

如果 smoke test 报错：

```text
TripoSR inference failed: expected scalar type Half but found Float
```

说明 TripoSR 已经开始真实推理，但全局 `TORCH_DTYPE=float16` 把模型移到了 half，而 TripoSR 上游预处理仍产生 float32 输入。新版 emoSVG wrapper 默认会让 TripoSR 保持 float32，这对 24 GB 显存的 4090 更稳。

请先同步新版：

```text
src/modules/reconstructor_3d/reconstructor.py
```

`.env` 中可以显式加入：

```bash
TRIPOSR_FORCE_FLOAT32=1
```

`1` 是新版默认值。只有在你确认 TripoSR 模型和输入路径都支持 half 推理时，才把它设成 `0`。

---

## 12. 继续阅读

完整服务器跑通手册见：

```text
SERVER_RUN_GUIDE.zh.md
```
