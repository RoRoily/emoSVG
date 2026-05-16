# animeFaceDetector 环境搭建指南

## 重要更新：独立环境现在可以被 emoSVG 直接调用

当前代码已经接入 `external_anime_face_detector` 后端：主环境 `EmoSVG` 不需要安装 OpenMMLab，也不需要直接 `import anime_face_detector`；它会通过子进程调用独立环境 `animeFaceDetector` 里的 Python，执行 `scripts/anime_face_detect.py`，再把 bbox/keypoints JSON 转回 emoSVG 的 `CartoonFaceGeometry`。

在主环境 `EmoSVG` 中追加或导出这些变量：

```bash
export CARTOON_ANALYZER_BACKEND=external_anime_face_detector
export CARTOON_ANALYZER_EXTERNAL_PYTHON=/data3/zhengmuhan/.conda/envs/animeFaceDetector/bin/python
export CARTOON_ANALYZER_EXTERNAL_SCRIPT=/data3/zhengmuhan/workspace/emoSVG/scripts/anime_face_detect.py
export CARTOON_ANALYZER_DEVICE=cpu
export CARTOON_ANALYZER_ALLOW_HEURISTIC_FALLBACK=0
```

验证命令：

```bash
conda activate EmoSVG
cd /data3/zhengmuhan/workspace/emoSVG
set -a
source .env
set +a

python scripts/cartoon_diagnostics.py \
  /data3/zhengmuhan/workspace/emoSVG/test_data/1.png \
  --output-root outputs/debug_stage012_external \
  --backend external_anime_face_detector \
  --external-python /data3/zhengmuhan/.conda/envs/animeFaceDetector/bin/python \
  --device cpu \
  --strict-trained
```

如果输出里的 `backend_used` 是 `external_anime_face_detector`，说明 emoSVG 主环境已经成功调用独立 `animeFaceDetector` 环境。之后启动 API 时也只需要确保 `.env` 已加载，`/animate` 的 `cartoon_rig` 路线就会走训练版动漫脸关键点检测。

本文记录为 emoSVG 的 stage 0/1/2 引入训练版动漫脸检测器 `hysts/anime-face-detector` 时，独立环境 `animeFaceDetector` 的正确搭建方法、踩坑记录和验证步骤。

## 1. 为什么要单独建环境

`hysts/anime-face-detector` 依赖较老的 OpenMMLab 技术栈：

```text
mmcv-full 1.x
mmdet 2.x
mmpose 0.x
```

而 emoSVG 主环境目前使用较新的 PyTorch / CUDA / transformers / opencv 组合。如果把 OpenMMLab 老栈直接装进 `EmoSVG` 主环境，容易出现：

```text
torch / torchvision 被替换
numpy / protobuf / rich 被降级或冲突
opencv-contrib-python 被装回
mmcv-full 编译失败
mmdet / mmpose 版本装错
```

因此推荐：

```text
EmoSVG              # 主项目环境，跑 emoSVG API、cartoon_rig、LivePortrait 等
animeFaceDetector   # 独立环境，只负责 anime-face-detector 检测
```

当前 emoSVG 代码已经支持 `anime_face_detector` 后端，但如果检测器只安装在独立环境里，主环境暂时不能直接 import。后续最稳的集成方式是：emoSVG 主环境通过子进程或小服务调用 `animeFaceDetector` 环境。

## 2. 正确版本组合

本次验证过程里，推荐使用下面这组版本：

```text
python              3.8
torch               1.13.0+cu117
torchvision         0.14.0+cu117
mmcv-full           1.7.0
mmdet               2.28.2
mmpose              0.29.0
anime-face-detector 0.0.9
```

关键点：

- `mmpose==0.29.0` 要求 `mmcv <= 1.7.0`。
- `mmcv-full==1.7.2` 不兼容，会报 `Please install mmcv>=1.3.8, <=1.7.0`。
- `mmcv-full` 必须安装预编译 wheel，不能让它源码编译。

## 3. 从零搭建环境

如果当前 `animeFaceDetector` 环境已经被装乱，建议直接重建：

```bash
conda deactivate
conda remove -n animeFaceDetector --all -y
conda create -n animeFaceDetector python=3.8 -y
conda activate animeFaceDetector

python -m pip install --upgrade pip setuptools wheel
```

安装 PyTorch 旧栈：

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

## 4. 安装 mmcv-full 1.7.0

必须指定 OpenMMLab wheel 地址：

```bash
pip install mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
```

正确日志应该出现 `.whl`：

```text
Downloading ... mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl
Successfully installed mmcv-full-1.7.0
```

如果看到：

```text
Downloading ... mmcv-full-1.7.0.tar.gz
Building wheel for mmcv-full
error: command '/usr/bin/gcc' failed with exit code 1
```

说明没有拿到预编译 wheel，应该立刻停止，不要让它源码编译。

## 5. 安装 mmdet / mmpose / anime-face-detector

```bash
pip install mmdet==2.28.2 mmpose==0.29.0 anime-face-detector \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证版本：

```bash
python - <<'PY'
import torch, mmcv, mmdet, mmpose
print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)
PY
```

期望：

```text
mmcv: 1.7.0
mmdet: 2.28.2
mmpose: 0.29.0
```

## 6. 下载模型权重

`anime-face-detector` 需要两个权重文件：

```text
mmdet_anime-face_yolov3.pth      # 人脸检测，约 235MB
mmpose_anime-face_hrnetv2.pth    # 28点landmark，约 38MB
```

默认下载地址来自 GitHub Release：

```text
https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth
https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth
```

默认缓存目录是：

```bash
python - <<'PY'
import torch
print(torch.hub.get_dir())
PY
```

权重应放在：

```text
$(torch.hub.get_dir())/checkpoints/
```

创建缓存目录：

```bash
CACHE_DIR=$(python - <<'PY'
import torch
print(torch.hub.get_dir())
PY
)

mkdir -p "$CACHE_DIR/checkpoints"
cd "$CACHE_DIR/checkpoints"
```

如果自动下载很慢或断开，手动下载。

检测权重：

```bash
wget -c "https://gh-proxy.com/https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth" \
  -O mmdet_anime-face_yolov3.pth
```

landmark 权重：

```bash
wget -c "https://gh-proxy.com/https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth" \
  -O mmpose_anime-face_hrnetv2.pth
```

如果 `gh-proxy.com` 不可用，可以换：

```bash
wget -c "https://ghproxy.net/https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth" \
  -O mmpose_anime-face_hrnetv2.pth
```

或：

```bash
wget -c "https://github.akams.cn/https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth" \
  -O mmpose_anime-face_hrnetv2.pth
```

确认权重：

```bash
ls -lh "$CACHE_DIR/checkpoints"/mmdet_anime-face_yolov3.pth
ls -lh "$CACHE_DIR/checkpoints"/mmpose_anime-face_hrnetv2.pth
```

期望大小：

```text
mmdet_anime-face_yolov3.pth      235M
mmpose_anime-face_hrnetv2.pth    38M
```

## 7. 验证 detector

先用 CPU 验证，避免 4090 + torch1.13/cu117 可能出现 CUDA 架构问题：

```bash
conda activate animeFaceDetector
cd ~/workspace/emoSVG

python - <<'PY'
from anime_face_detector import create_detector

detector = create_detector("yolov3", device="cpu")
print("anime-face-detector CPU OK")
PY
```

如果 CPU OK，再尝试 GPU：

```bash
python - <<'PY'
from anime_face_detector import create_detector

detector = create_detector("yolov3", device="cuda:0")
print("anime-face-detector CUDA OK")
PY
```

如果 GPU 失败，stage 0/1/2 单图诊断可以先用 CPU。

## 8. 跑一张图确认 28 点输出

```bash
python - <<'PY'
import cv2
from anime_face_detector import create_detector

img = cv2.imread("/data3/zhengmuhan/workspace/emoSVG/test_data/1.png")
detector = create_detector("yolov3", device="cpu")
preds = detector(img)

print("num faces:", len(preds))
for i, p in enumerate(preds):
    print("face", i)
    print("bbox:", p["bbox"])
    print("keypoints shape:", p["keypoints"].shape)
    print("mouth points:", p["keypoints"][23:28])
PY
```

期望：

```text
num faces: 1
keypoints shape: (28, 3)
```

如果输出这个结果，说明 `animeFaceDetector` 独立环境已经跑通。

## 9. 已遇到的问题与解决方法

### 9.1 不要装进 EmoSVG 主环境

问题：

```text
mediapipe requires opencv-contrib-python
open-clip-torch requires timm
protobuf / rich / typer 冲突
mmcv-full build failed
chumpy build failed
```

原因：

OpenMMLab 老栈会改动主环境依赖，和 emoSVG 当前依赖不兼容。

解决：

使用独立环境 `animeFaceDetector`。

### 9.2 `No module named 'mmengine'`

错误：

```text
ModuleNotFoundError: No module named 'mmengine'
```

原因：

装成了 `mmdet 3.x` 或 `mmpose 1.x` 新体系。`anime-face-detector` 需要旧体系。

解决：

```bash
pip uninstall mmdet mmpose mmengine -y
pip install mmdet==2.28.2 mmpose==0.29.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 9.3 `MMCV==1.7.2 is used but incompatible`

错误：

```text
AssertionError: MMCV==1.7.2 is used but incompatible.
Please install mmcv>=1.3.8, <=1.7.0.
```

原因：

`mmpose==0.29.0` 不接受 `mmcv 1.7.2`。

解决：

```bash
pip uninstall mmcv mmcv-full -y
pip install mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
```

### 9.4 `Failed building wheel for mmcv-full`

错误：

```text
error: command '/usr/bin/gcc' failed with exit code 1
Failed building wheel for mmcv-full
```

原因：

pip 没拿到预编译 wheel，开始源码编译。

解决：

必须指定 wheel 地址：

```bash
pip install mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
```

安装日志必须出现：

```text
mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl
```

### 9.5 权重下载很慢或断开

现象：

```text
235M 下载很慢
RemoteDisconnected: Remote end closed connection without response
```

原因：

默认权重来自 GitHub Release，服务器访问 GitHub 不稳定。

解决：

手动下载到 torch hub cache：

```bash
CACHE_DIR=$(python - <<'PY'
import torch
print(torch.hub.get_dir())
PY
)

mkdir -p "$CACHE_DIR/checkpoints"
cd "$CACHE_DIR/checkpoints"

wget -c "https://gh-proxy.com/https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth" \
  -O mmpose_anime-face_hrnetv2.pth
```

### 9.6 能不能从 Hugging Face 下载

目前官方 `hysts/anime-face-detector` 的 Hugging Face Space 主要托管 demo 代码，不直接托管这两个 `.pth` 权重。官方包里的权重 URL 仍指向 GitHub Release。

所以最快方案仍是 GitHub Release + 代理，或者本地下载后 `scp` 到：

```text
~/.cache/torch/hub/checkpoints/
```

文件名必须完全一致。

## 10. 和 emoSVG 主环境的关系

当前两个环境的职责：

```text
EmoSVG
  - 主项目环境
  - FastAPI
  - cartoon_rig
  - LivePortrait / ToonCrafter / TripoSR / SAM

animeFaceDetector
  - 独立动漫脸检测环境
  - hysts/anime-face-detector
  - OpenMMLab 旧栈
```

当前 emoSVG 代码中已有直接 import 后端：

```bash
python scripts/cartoon_diagnostics.py image.png \
  --backend anime_face_detector \
  --device cuda:0 \
  --strict-trained
```

但这个命令要求 `anime-face-detector` 安装在当前 Python 环境里。由于我们把它放在独立环境，下一步推荐改成：

```text
EmoSVG 主环境
  -> 调用 animeFaceDetector 环境里的检测脚本
  -> 检测脚本输出 bbox/keypoints JSON
  -> emoSVG 读取 JSON 并转成 CartoonFaceGeometry
```

这样既能使用训练模型，又不会污染 emoSVG 主环境。

## 11. 最终成功标准

满足以下条件即可认为 `animeFaceDetector` 环境搭建成功：

```text
1. conda activate animeFaceDetector 成功
2. import torch/mmcv/mmdet/mmpose 成功
3. mmcv == 1.7.0
4. mmdet == 2.28.2
5. mmpose == 0.29.0
6. 两个权重文件都在 torch hub checkpoints 目录
7. create_detector("yolov3", device="cpu") 成功
8. 测试图输出 num faces >= 1
9. keypoints shape == (28, 3)
```
