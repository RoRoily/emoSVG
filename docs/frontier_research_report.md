# emoSVG 前沿相关研究与工程启示报告

生成日期：2026-05-10  
调研范围：角色动画、图生视频/首末帧视频、卡通插帧、单图 3D、SVG 矢量化、生成式评估 benchmark。  
调研原则：优先采用 CVPR、ICCV、ECCV、ICLR、NeurIPS、ACM TOG/SIGGRAPH 等顶会/顶刊论文与官方项目页；对 Wan、LTX、Hunyuan3D 等强工程开源项目单独标注为“工程前沿/工业开源”，不把它们伪装成已同行评审成果。

---

## 0. 结论摘要

emoSVG 当前不是“某个单模型的 demo”，而是一个单图 IP 角色到多资产输出的工程管线：

```text
源角色图像
  -> CLIP/IP 特征
  -> LivePortrait / ToonCrafter 生成 meme 动画
  -> 可选 TripoSR 单图 3D
  -> 可选 SAM + Bezier SVG 分层矢量化
```

结合 `docs/architecture.md` 与代码现状，项目最值得继续推进的方向是：

1. **先把现有 claim 变成可验证的基线**  
   `AnimationRequest` 已有 `use_toon_crafter_as_driver` 和 `driver_num_frames`，但 `FullPipelineRequest`、`/animate`、`/generate` 没有透传。架构里最重要的 ToonCrafter driver mode 目前对主流程不可达。这是 P0。

2. **不要急着替换模型，先建立评估 harness**  
   当前测试主要验证 fallback、shape、文件产出和 API 路由，不验证真实视觉质量。应借鉴 VBench、EvalCrafter、GPTEval3D、Eval3D 的思路，把视频、3D、SVG 拆成可诊断维度：身份保持、动作夸张度、时间一致性、几何质量、SVG 可编辑性、资源成本。

3. **动画路线：LivePortrait 负责低延迟可控，ToonCrafter/FLF2V 负责超出生理表情空间的夸张运动**  
   LivePortrait 的隐式关键点路径适合快速生成和可控调参；ToonCrafter、SEINE、Wan FLF2V 等首末帧/过渡生成路径更适合 “source -> peak” 边界帧驱动的 meme 动画。

4. **3D 路线：TripoSR 可保留为 fast mode，但应抽象后端**  
   LRM/TripoSR 证明了单图前馈 3D 的低延迟可行性；SF3D、TRELLIS、Hunyuan3D 2.0 说明高质量资产正在向 UV、PBR、多表示、局部编辑发展。emoSVG 应把 TripoSR 作为 fast backend，预留 SF3D/TRELLIS/Hunyuan3D 等 quality backend。

5. **SVG 路线：SAM + Bezier 是原型，不是终点**  
   DiffVG、Im2Vec、VectorFusion、SVGDreamer、Layered Image Vectorization 的共同启示是：好的 SVG 不是“尽可能贴合像素轮廓”，而是“语义分层、紧凑、可编辑、可渐进细化”。emoSVG 应加 VTracer 作为工程 baseline，再引入语义层级与 raster-loss refinement。

---

## 1. 本 repo 当前状态复核

### 1.1 已有架构优势

- `ModelRegistry` 统一管理模型生命周期和 VRAM，适合多重模型串联。
- 每个模块都有 CPU fallback，因此 CI 可以在没有 GPU 和权重时跑通接口契约。
- 模块边界清晰：`IPExtractor`、`MemeAnimator`、`Reconstructor3D`、`SVGVectorizer`。
- `MemeAnimator` 内部已经设计了 ToonCrafter 双模式：
  - interpolation mode：LivePortrait 渲染完整 keyframe 序列，ToonCrafter 插帧；
  - driver mode：LivePortrait 只渲染 peak frame，ToonCrafter 从 source/peak 生成完整动画。
- 3D/SVG 支持使用原图或 peak keyframe 作为输入，这对“标准资产”和“表情资产”两类需求都有意义。

### 1.2 主要工程缺口

| 缺口 | 证据 | 影响 | 优先级 |
|---|---|---|---|
| ToonCrafter driver mode 主流程不可达 | `src/modules/meme_animator/schemas.py` 有 `use_toon_crafter_as_driver`，但 `src/pipeline/full_pipeline.py`、`src/api/routers/animate.py`、`src/api/routers/generate.py` 没有透传 | 架构文档中最关键的“绕过 LivePortrait 人脸空间限制”的路径无法通过 API 使用 | P0 |
| ToonCrafter adapter 可能与真实项目结构不匹配 | 代码期待 `tooncrafter.inference.ToonCrafterInference`；跑通记录里实际验证的是 `import lvdm` | 真实权重路径可能无法执行，只在 fallback/mock 中“看起来可用” | P0 |
| `ip_adapter_embeds` 对 LivePortrait 可能无效 | LivePortrait 官方 API 未必接收该 kwarg；代码中属于乐观兼容写法 | 身份条件可能被静默忽略，评估时会误判“IP 特征已生效” | P0 |
| 测试不验证视觉质量 | 当前单元/集成测试主要验证 fallback 和文件存在 | 无法判断动画、SVG、3D 是否真正变好 | P0 |
| 文档与代码漂移 | 架构文档写 SAM VRAM 约 2.4GB，代码 `src/modules/svg_vectorizer/segmentor.py` 中 `_SAM_VRAM_GB = 6.5` | VRAM 预算与调度判断可能失真 | P1 |
| peak frame 选择不稳 | `FullPipeline` 用 `animation.keyframes[len(... ) // 2]`，而 keyframes 是每 5 帧采样后的 metadata | smoothing/driver 后不一定是真正 peak | P1 |

---

## 2. 技术地图

```text
emoSVG 目标：单张 IP 图像 -> 动画 + 3D + SVG

一、身份/外观保持
  CLIP / DINOv2 / ArcFace / IP-Adapter / PhotoMaker / DreamBooth
  关注：外观一致性、角色可识别性、跨帧漂移

二、角色动画
  FOMM / TPSMM / LivePortrait / Animate Anyone / MagicAnimate / Champ
  关注：可控运动、时间一致性、身份保持、非人脸角色泛化

三、图生视频与首末帧过渡
  ToonCrafter / DynamiCrafter / SEINE / Stable Video Diffusion / Wan FLF2V / LTX
  关注：source -> peak 过渡、非线性运动、遮挡显露、扩散视频先验

四、单图 3D
  LRM / TripoSR / Wonder3D / MVDream / SyncDreamer / DreamGaussian / SF3D / TRELLIS / Hunyuan3D
  关注：低延迟 vs 高质量、mesh/texture/UV/PBR、多视角一致性

五、SVG 矢量化
  SAM / SAM2 / VTracer / DiffVG / DeepSVG / Im2Vec / VectorFusion / SVGDreamer / Layered Vectorization
  关注：语义分层、紧凑性、可编辑性、raster fidelity

六、评估
  VBench / EvalCrafter / FVD / LPIPS / RAFT / ArcFace / GPTEval3D / Eval3D
  关注：不要用单一分数；用任务化、诊断型指标
```

---

## 3. 代表研究与开源项目

### 3.1 身份与参考图条件

| 工作 | 年份/来源 | 核心贡献 | 对 emoSVG 的启示 |
|---|---|---|---|
| CLIP: Learning Transferable Visual Models From Natural Language Supervision | ICML 2021，OpenAI 官方页：https://openai.com/research/clip，ICML 页：https://icml.cc/virtual/2021/oral/9194 | 用大规模图文对比学习得到可迁移视觉语义特征 | 可作为身份/语义相似度的粗指标；但 CLIP 更偏语义，不足以评估细粒度角色身份 |
| ArcFace | CVPR 2019，论文页：https://huggingface.co/papers/1801.07698 | 人脸识别中的角度 margin 特征，适合身份相似度 | 对“类人脸 IP”可用；对非人脸 mascot 不可靠，应和 DINO/CLIP/颜色/mask 指标组合 |
| DINOv2 | 2023，论文页：https://huggingface.co/papers/2304.07193 | 自监督视觉 foundation feature，在密集匹配、实例检索、分割特征上强 | 比 CLIP 更适合做角色视觉一致性、局部 patch 相似度、SVG/3D 多视角 render 对齐 |
| IP-Adapter | 2023，论文页：https://huggingface.co/papers/2308.06721 | 用解耦 cross-attention 将 image prompt 接入 diffusion | emoSVG 现在只提取 CLIP embedding，还没有真正训练/验证 IP 条件路径。若未来接视频扩散后端，IP-Adapter 思路更合适 |
| DreamBooth | CVPR 2023，CVPR 页：https://cvpr.thecvf.com/virtual/2023/poster/23180 | 少量图像 fine-tune，让模型绑定特定 subject | 不适合默认实时路径，但可作为高质量离线 personalized backend |
| PhotoMaker | CVPR 2024，CVPR 页：https://cvpr.thecvf.com/virtual/2024/poster/30805，OpenAccess：https://openaccess.thecvf.com/content/CVPR2024/html/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.html | stacked ID embedding，兼顾速度、身份与文本控制 | 启发 emoSVG 对“多参考图 IP 设定”的扩展：单张源图不够时，可引入多图 ID embedding |
| InstantID | 2024，论文页：https://huggingface.co/papers/2401.07519 | 单张脸图快速身份保持生成 | 对人脸角色有价值；对非人脸 IP 仍需要通用视觉/分割特征 |

**关键判断**：  
emoSVG 现在的 `IPExtractor` 更像“提取一个 CLIP 图像 embedding 并传给后续模块”，还不是完整的身份保持机制。若目标是 IP 角色一致性，评估层需要同时使用：

- CLIP/DINOv2 图像相似度；
- ArcFace/InsightFace，仅限类人脸；
- 前景 mask IoU 与颜色分布漂移；
- 局部关键区域，例如眼睛、嘴、logo、轮廓线。

---

### 3.2 角色动画与肖像动画

| 工作 | 年份/来源 | 核心贡献 | 对 emoSVG 的启示 |
|---|---|---|---|
| First Order Motion Model | NeurIPS 2019，论文页：https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-a | 自监督关键点 + 局部仿射运动，将 source image 按 driving video 动起来 | 是 LivePortrait/FOMM 系路线的经典基础；说明 keypoint/warp 路线可控、轻量，但大形变能力有限 |
| Thin-Plate Spline Motion Model | CVPR 2022，论文页：https://huggingface.co/papers/2203.14367 | TPS motion estimation + occlusion mask，改善大 pose gap 的任意物体动画 | 对非人脸 IP 更有启发：可以引入 TPS/mask-aware deformation，而不是只调人脸 expression coeff |
| LivePortrait | 2024，论文页：https://huggingface.co/papers/2407.03168，项目页：https://liveportrait.github.io/ | 隐式关键点肖像动画，强调效率、可控性、泛化；官方摘要报告 RTX 4090 上可达到很低延迟 | 很适合作为 emoSVG 默认 fast animation backend；但它的人脸表达空间不适合极端 cartoon squash-and-stretch |
| Animate Anyone | CVPR 2024，CVPR 页：https://cvpr.thecvf.com/virtual/2024/poster/31222 | ReferenceNet + pose guider + temporal modeling，用 diffusion 做角色动画 | 对完整人物/角色舞蹈更强；启发 emoSVG 增加“参考姿态/动作模板”输入 |
| MagicAnimate | CVPR 2024，CVPR 页：https://cvpr.thecvf.com/virtual/2024/poster/29797，项目页：https://www.magicanimate.org/ | diffusion + appearance encoder + temporal modeling，提升人像动画时间一致性和身份保持 | 对“时序一致性 + 参考图细节保真”有借鉴；但成本高于 LivePortrait |
| Champ | 2024，论文页：https://huggingface.co/papers/2403.14781 | 用 3D parametric human guidance、depth/normal/semantic map 控制人体动画 | 启发：如果 emoSVG 未来扩展全身 IP，可用 3D/normal/semantic 控制替代纯 facial coeff |
| AniPortrait / EMO | 2024，AniPortrait：https://huggingface.co/papers/2403.17694，EMO：https://huggingface.co/papers/2402.17485 | 音频驱动肖像动画，重视表情自然度和身份一致性 | 不是当前主线，但可作为未来“音频 meme 表情”扩展参考 |

**对 emoSVG 的直接启示**：

1. LivePortrait 应保持为默认低成本路径，但不要把它当作 cartoon 形变的最终表达空间。
2. 对极端 meme 表情，应走 `source frame + peak frame -> video transition` 的扩散视频路径。
3. 非人脸 IP 需要 mask-aware / TPS / contour-aware deformation。仅靠 21 个隐式人脸关键点会导致：
   - 无眼/无嘴角色无法表达；
   - mascot/emoji 的可动部位错配；
   - 大幅 squash/stretch 出现人脸网格 artifact。

---

### 3.3 图生视频、首末帧视频与卡通插帧

| 工作 | 年份/来源 | 核心贡献 | 对 emoSVG 的启示 |
|---|---|---|---|
| ToonCrafter | ACM TOG 2024，Monash 页面：https://research.monash.edu/en/publications/tooncrafter-generative-cartoon-interpolation/，论文页：https://huggingface.co/papers/2405.17933 | 面向卡通视频插帧，用生成式框架处理非线性大运动、遮挡/显露、cartoon domain gap | 与 emoSVG 的 source/peak 设计高度契合。应作为 driver mode 的首个真实验证对象 |
| DynamiCrafter | ECCV 2024 Oral，ECCV 页：https://eccv.ecva.net/virtual/2024/oral/1246 | 将开放域静图动画化，利用 text-to-video diffusion motion prior，并加入图像条件 | 可作为“通用图生视频 backend”候选，不限 cartoon；但动作可控性可能弱于首末帧路径 |
| SEINE | ICLR 2024，论文页：https://proceedings.iclr.cc/paper_files/paper/2024/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html | 随机 mask 视频扩散，用于 generative transition/prediction，可由不同 scene/image 生成过渡 | 对 emoSVG 的 driver mode 很关键：source/peak 可看作一个短 transition 问题 |
| Stable Video Diffusion | 2023，Stability AI 页：https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets，论文页：https://huggingface.co/papers/2311.15127 | 大规模 latent video diffusion，支持 image-to-video | 可作为通用 I2V baseline，但对“指定 peak frame”的控制不如 FLF2V/transition 模型 |
| Wan2.1 FLF2V | 工程前沿，GitHub：https://github.com/Wan-Video/Wan2.1 | First-Last-Frame-to-Video，14B 720P，用 first/last frame + prompt 生成过渡 | 与 emoSVG source/peak 边界帧完全对齐；但模型大，适合作 quality/premium backend |
| Wan2.2 / Wan2.2-Animate | 工程前沿，GitHub：https://github.com/Wan-Video/Wan2.2 | I2V、TI2V、Animate 等大型视频生成能力；官方示例提示 I2V A14B 需要很高 VRAM | 不适合作默认 10GB 路径；可作为外部服务或离线高质量后端 |
| LTX-Video / LTX-2 | 工程前沿，GitHub：https://github.com/Lightricks/LTX-Video，文档：https://docs.ltx.video/open-source-model/usage-guides/image-to-video | 支持 I2V、多关键帧、视频延展、音视频生成等生产型能力 | 可作为多关键帧动画 backend 候选；但部署成本高，需要与 emoSVG 的 VRAM 管理策略隔离 |

**关键判断**：

- emoSVG 的 ToonCrafter driver mode 是正确方向，但现在工程上还没有闭环。
- 首末帧/transition 模型比普通 I2V 更适合 meme 动画，因为 emoSVG 已经能生成一个 peak expression frame。
- 大模型视频 backend 不应该直接塞进当前单进程 10GB VRAM 预算。更合理的是：
  - 本地 fast path：LivePortrait / ToonCrafter；
  - 本地或服务器 quality path：Wan FLF2V / LTX multi-keyframe；
  - 统一接口：`VideoBackend.generate(source, peak, prompt, num_frames, seed)`。

---

### 3.4 单图 3D 与 3D 资产生成

| 工作 | 年份/来源 | 核心贡献 | 对 emoSVG 的启示 |
|---|---|---|---|
| LRM | ICLR 2024，ICLR 页：https://proceedings.iclr.cc/paper_files/paper/2024/hash/dcad3425f5c8c36b5b3885c091bf1257-Abstract-Conference.html，Adobe 页：https://research.adobe.com/publication/lrm-large-reconstruction-model-for-single-image-to-3d/ | 500M transformer，从单图直接预测 NeRF；大规模多视角训练 | TripoSR 的上游思想基础；说明前馈单图 3D 是合理产品路径 |
| TripoSR | 2024，论文页：https://huggingface.co/papers/2403.02151 | 基于 LRM 的快速单图 3D，报告可在约 0.5s 量级生成 mesh | 适合作 emoSVG fast mode；但 texture/UV/PBR 不是强项 |
| Wonder3D | CVPR 2024，CVPR 页：https://cvpr.thecvf.com/virtual/2024/poster/29642 | 单图生成多视角 normal/color，再通过 normal fusion 得到高质量 textured mesh | 对高质量离线 reconstruction 有启发，但 2-3 分钟级别不适合作默认实时路径 |
| MVDream | ICLR 2024，ICLR 页：https://proceedings.iclr.cc/paper_files/paper/2024/hash/adbe936993aa7cf41e45054d8b72f183-Abstract-Conference.html | multi-view diffusion 作为 3D prior，改善多视角一致性 | 对 text/image-to-3D 的一致性有价值，可作为未来 high-quality backend 的思想参考 |
| SyncDreamer | ICLR 2024 Spotlight，ICLR 页：https://iclr.cc/virtual/2024/poster/18828 | 从单视图生成多视角一致图像，服务 novel view / image-to-3D | 可用于给 TripoSR/SVG 之外的 3D quality path 补多视角输入 |
| DreamGaussian | ICLR 2024，ICLR 页：https://proceedings.iclr.cc/paper_files/paper/2024/hash/905202e21386913d8eac637c2b50f590-Abstract-Conference.html | 用 3D Gaussian Splatting 加速 3D 内容生成，并转 mesh/refine texture | 对“速度-质量折中”的 3D 资产后端有参考 |
| SF3D / Stable Fast 3D | CVPR 2025，项目页：https://stable-fast-3d.github.io/，Stability AI：https://stability.ai/research/sf3d-stable-fast-3d-mesh-reconstruction-with-uv-unwrapping-and-illumination-disentanglement | 单图快速生成 UV-unwrapped textured mesh，并预测 material/normal、去光照 | 对 emoSVG 价值极高：比 TripoSR 更接近生产资产，适合作下一个 3D backend 候选 |
| TRELLIS | CVPR 2025，Microsoft 页：https://www.microsoft.com/en-us/research/publication/structured-3d-latents-for-scalable-and-versatile-3d-generation/，GitHub：https://github.com/microsoft/TRELLIS | SLAT 统一结构 latent，可输出 Radiance Fields、3D Gaussians、meshes，支持局部编辑 | 可作为 quality backend/研究标杆；对“多输出资产统一表示”尤其重要 |
| Hunyuan3D 2.0 | 2025，论文页：https://huggingface.co/papers/2501.12202，GitHub：https://github.com/Tencent-Hunyuan/Hunyuan3D-2 | shape + texture 两阶段，高分辨率 textured 3D asset | 工程上成熟度高，适合对标 quality backend；但资源需求与部署复杂度需单独管理 |
| MeshAnything | 2024，论文页：https://huggingface.co/papers/2406.10163 | 生成 artist-created mesh，降低面数，改善 mesh 可用性 | 对 emoSVG 的 mesh post-process 很有启发：不仅要“重建出来”，还要“可用、低面数、拓扑友好” |

**对 emoSVG 的直接启示**：

1. `Reconstructor3D` 应变成 backend 抽象：

```text
ReconstructionBackend
  - TripoSRBackend       # fast / current
  - SF3DBackend          # fast textured mesh + UV/PBR
  - TRELLISBackend       # quality / multi-representation
  - Hunyuan3DBackend     # quality / textured asset
```

2. 输出指标不应只看 OBJ/GLB 是否存在，应增加：
   - multi-view render similarity；
   - mesh manifold/watertight；
   - nondegenerate faces；
   - vertex/face count；
   - UV/material 是否存在；
   - texture atlas 是否合理；
   - GLB validator 结果。

3. `use_source_for_3d_svg` 应继续保留，但建议拆成更明确的产品选项：
   - `asset_pose="canonical"`：用于游戏/贴纸/商品资产；
   - `asset_pose="expressive_peak"`：用于 meme 表情资产；
   - `asset_pose="both"`：同时输出两套。

---

### 3.5 SVG 矢量化与语义分层

| 工作 | 年份/来源 | 核心贡献 | 对 emoSVG 的启示 |
|---|---|---|---|
| VTracer | 工程开源，GitHub：https://github.com/visioncortex/vtracer，文档：https://www.visioncortex.org/vtracer-docs | Rust 实现的 raster-to-vector，支持彩色图，输出紧凑 SVG | 应作为 emoSVG SVG 模块的 production baseline，与当前 SAM+Bezier 做质量/大小对比 |
| SAM | ICCV 2023，论文页：https://huggingface.co/papers/2304.02643 | promptable segmentation，11M 图像、10 亿 mask 数据集，零样本分割强 | 当前 SVG segmentation 的基础合理；但 automatic masks 不等于可编辑语义层 |
| SAM 2 | 2024，论文页：https://huggingface.co/papers/2408.00714，Meta 介绍：https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/ | 图像+视频统一分割，streaming memory，可跨帧跟踪对象 | 对动画帧级 SVG/层跟踪很重要，可让不同帧共享 layer id |
| DiffVG | ACM TOG 2020，论文信息：https://colab.ws/articles/10.1145/3414685.3417871 | 可微矢量图 rasterizer，把 raster loss 用于 SVG 优化 | emoSVG 的 Bezier 拟合后可加 DiffVG-style refine，减少 contour 噪声 |
| DeepSVG | NeurIPS 2020，NeurIPS 页：https://proceedings.neurips.cc/paper/2020/hash/bcf9d6bd14a2095866ce8c950b702341-Abstract.html | 分层生成 SVG icon，支持 latent interpolation/animation | 对“SVG 动画/可编辑 icon latent”有长期价值，但需要 SVG 数据训练 |
| Im2Vec | CVPR 2021 Oral，项目页：https://geometry.cs.ucl.ac.uk/projects/2021/im2vec/ | 无需 vector supervision，仅用 raster supervision 学 SVG 生成 | 对 emoSVG 的“已有 raster，想要 vector”非常相关；可借鉴 differentiable rasterization 监督 |
| VectorFusion | CVPR 2023，CVPR 页：https://cvpr.thecvf.com/virtual/2023/poster/20940 | 用 diffusion + differentiable vector rasterizer 做 text-to-SVG | 说明扩散先验可优化 SVG 语义质量；但 text-to-SVG 不是 emoSVG 主需求 |
| SVGDreamer | CVPR 2024，CVPR 页：https://cvpr.thecvf.com/virtual/2024/poster/29501 | semantic-driven image vectorization + VPSD，提升可编辑性、视觉质量、多样性 | 强烈启发 emoSVG 的“前景/背景/对象层”分解 |
| Layered Image Vectorization via Semantic Simplification | CVPR 2025，CVPR 页：https://cvpr.thecvf.com/virtual/2025/poster/34467，项目页：https://szuviz.github.io/layered_vectorization/ | 渐进语义简化 + 两阶段 vectorization，得到紧凑、语义对齐的分层 SVG | 与 emoSVG 最匹配的前沿方向：从 SAM mask list 升级为语义层级 SVG |

**对 emoSVG 的直接启示**：

当前 `SVGVectorizer` 流程是：

```text
SAM/OpenCV masks -> contour -> Bezier -> <g> layers
```

它缺少三件事：

1. **语义层级**：现在是 mask 列表，不是 background/body/eyes/mouth/accessory/outlines 的可编辑结构。
2. **紧凑性约束**：Bezier tolerance 只能粗略控制节点数，缺少“文件大小/节点预算/视觉误差”的联合优化。
3. **raster-loss refinement**：拟合后没有根据 SVG rasterize 结果反向优化路径和颜色。

建议改成：

```text
SegmentationBackend:
  SAM / SAM2 / Grounded-SAM / OpenCV fallback

VectorizationBackend:
  current_sam_bezier
  vtracer_baseline
  layered_semantic_vectorization

Refinement:
  rasterize(SVG) -> compare(source) -> optimize color/path/node budget
```

---

### 3.6 生成式评估 benchmark

| 工作 | 年份/来源 | 核心贡献 | 对 emoSVG 的启示 |
|---|---|---|---|
| FVD | 2018/2019，论文页：https://huggingface.co/papers/1812.01717 | Fréchet Video Distance，用于视频生成分布质量 | 可作为参考，但对小样本、指定角色、指定动作不够诊断 |
| LPIPS | CVPR 2018，OpenAccess：https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html | 深特征感知距离，较 PSNR/SSIM 更符合人类感知 | 可用于 frame/source、SVG raster/source、多视角 render/source 的 perceptual error |
| RAFT | ECCV 2020，论文页：https://huggingface.co/papers/2003.12039 | 强 optical flow 估计 | 可用于 temporal smoothness、motion magnitude、flicker/flow consistency |
| VBench | CVPR 2024，CVPR 页：https://cvpr.thecvf.com/virtual/2024/poster/30230 | 把视频生成质量拆成分层维度，不只给单一分数 | emoSVG 评估应仿照它做“身份、动作、时间、画质、条件一致性”的拆解 |
| EvalCrafter | CVPR 2024，CVPR 页：https://cvpr.thecvf.com/virtual/2024/poster/29818，项目页：https://evalcrafter.github.io/ | 700 prompts、17 个客观指标、人类反馈拟合 leaderboard | 对 emoSVG 的报告/galleries/leaderboard 设计最直接 |
| VBench++ | 2024，论文页：https://huggingface.co/papers/2411.13503 | 更综合的视频生成 benchmark，细粒度指标 + 人类偏好 | 可作为未来扩展，但初期无需全量复刻 |
| DOVER | ICCV 2023，项目页：https://vqassessment.github.io/DOVER/ | 从 aesthetic 和 technical 角度做视频质量评价 | 可作为 animation output 的无参考视频质量指标 |
| GPTEval3D | CVPR 2024，项目页：https://gpteval3d.github.io/ | 用 GPT-4V 风格 VLM 判断 text-to-3D，与人类偏好对齐 | 对 3D 人工/自动混合评估有启发，可用多视角 render 给 VLM pairwise judge |
| Eval3D | CVPR 2025，CVPR 页：https://cvpr.thecvf.com/virtual/2025/poster/35203 | 可解释、细粒度 3D 生成评估，用多种模型/工具 probe 一致性 | 对 emoSVG 3D 评估非常适合：不要只看 CLIP，而要看几何/语义/多视角一致性 |

**关键判断**：  
emoSVG 不应追求一个“总分”。应按产物类型拆成诊断型指标：

```text
Animation:
  identity / meme intensity / temporal coherence / artifact / cost

3D:
  source alignment / geometry / texture / GLB usability / cost

SVG:
  raster fidelity / semantic layering / compactness / editability / cost
```

---

## 4. 与 emoSVG 当前方案的差距分析

### 4.1 动画模块差距

| 当前方案 | 前沿成熟做法 | 差距 | 建议 |
|---|---|---|---|
| LivePortrait 63-dim expression coeff 手写映射 | LivePortrait/FOMM/TPSMM 都依赖运动表示，但 diffusion 动画方法会显式建模时间与外观保持 | 手写映射缺少数据校准；非人脸 IP 适配弱 | 建立 expression calibration dataset，记录每个 coeff 的安全范围、artifact 边界 |
| ToonCrafter wrapper 假设 `generate/interpolate` API | ToonCrafter 官方项目是研究代码，未必提供封装好的 inference class | 真实推理路径高风险 | 做 `ToonCrafterBackend` 适配层，支持源码路径、ckpt、config、CLI fallback |
| driver mode 架构存在但 API 不可达 | SEINE/Wan FLF2V 都强调 transition/first-last-frame 控制 | 核心路径无法评估 | P0 透传 driver 参数 |
| IP embedding 直接传入 LivePortrait kwarg | IP-Adapter 是为 diffusion cross-attention 设计，不是 LivePortrait 原生条件 | 可能无效 | 先验证真实 API；若无效，把 IP features 用作评估/缓存，而不是宣称条件生效 |

### 4.2 3D 模块差距

| 当前方案 | 前沿成熟做法 | 差距 | 建议 |
|---|---|---|---|
| TripoSR -> mesh -> OBJ/GLB | SF3D 输出 UV-unwrapped textured mesh + material/normal；TRELLIS 多表示输出 | 当前资产生产性不足 | 增加 `fast`/`quality` backend |
| mesh stats 主要是顶点/面数 | Eval3D/GPTEval3D 强调多视角、几何、语义一致性 | 缺少质量判断 | 加 multi-view render gallery、watertight/manifold/normal/texture 指标 |
| peak keyframe 直接做 3D | 3D 模型通常需要 canonical view、清晰轮廓、少遮挡 | 表情夸张可能伤害 mesh | 输出 canonical 与 expressive 两套，或让用户选择 |

### 4.3 SVG 模块差距

| 当前方案 | 前沿成熟做法 | 差距 | 建议 |
|---|---|---|---|
| SAM automatic masks + Bezier | Layered Vectorization 强调语义简化、层级结构、紧凑 SVG | mask list 不等于可编辑 SVG | 加 semantic layer grouping |
| fallback OpenCV K-means contours | VTracer 已是成熟工程 baseline | fallback 质量/紧凑性可能低 | 引入 VTracer 作为默认 CPU baseline |
| Bezier fitting 后直接输出 | DiffVG/Im2Vec 用 raster loss 优化 | 没有闭环优化 | 增加 rasterize-evaluate-refine 阶段 |
| 静态 SAM | SAM2 支持视频对象跟踪 | 动画帧 SVG 无法稳定 layer id | 若做 animated SVG，采用 SAM2/video mask propagation |

---

## 5. 可落地评估方案

### 5.1 评测数据集

建议创建 `eval_assets/manifest.jsonl`，每条记录包含：

```json
{
  "id": "mascot_001",
  "source": "eval_assets/images/mascot_001.png",
  "category": "flat_mascot",
  "has_face": false,
  "has_alpha": true,
  "has_svg_gt": true,
  "has_mesh_gt": false,
  "expected_regions": ["head", "eyes", "mouth", "body"],
  "notes": "large eyes, simple flat colors"
}
```

建议首批 100 张：

| 类型 | 数量 | 用途 |
|---|---:|---|
| flat mascot/sticker | 30 | 主目标场景 |
| anime/chibi/emoji-like | 20 | 卡通夸张场景 |
| face-like IP | 20 | ArcFace/InsightFace 可用 |
| synthetic SVG-known | 15 | SVG ground truth |
| 3D-known rendered assets | 10 | 3D ground truth |
| adversarial cases | 5 | 细线、文字、多脸、透明孔洞、复杂背景 |

### 5.2 动画指标

| 维度 | 指标 | 实现建议 |
|---|---|---|
| 身份保持 | CLIP/DINOv2 similarity，face-like 样本用 ArcFace | source vs 每帧/首末帧 |
| 颜色保持 | 前景区域 Lab color histogram distance | 需要 foreground mask |
| 时间一致性 | temporal LPIPS、flow warping error、flicker score | RAFT 或轻量 optical flow |
| meme 表现力 | silhouette height/width change、mouth/eye region area change、peak displacement | 由 mask/landmark/区域检测近似 |
| 动作曲线 | wind-up、attack、hold、settle 是否符合预设帧数和强度 | 对 `SquashParams` 与输出帧同时打点 |
| artifact | 出框、空洞、边缘拖影、突然变色 | 自动规则 + 人工 review |
| 成本 | latency、peak VRAM、输出大小 | 与 `ModelRegistry` 状态联动 |

### 5.3 3D 指标

| 维度 | 指标 |
|---|---|
| 几何有效性 | vertex/face count、nondegenerate face ratio、watertight、manifold、自相交 |
| 源图对齐 | 多视角 render 与 source 的 CLIP/DINO/LPIPS；正面 render 与 source 的 mask IoU |
| 纹理/材质 | 是否有 UV、texture atlas、PBR/material、normal map |
| 生产可用性 | GLB validator、文件大小、加载时间、单位尺度、朝向 |
| 若有 GT | Chamfer distance、F-score、normal consistency |

### 5.4 SVG 指标

| 维度 | 指标 |
|---|---|
| 视觉拟合 | rasterized SVG vs source 的 PSNR/SSIM/LPIPS |
| 紧凑性 | path count、node count、file size、render time |
| 可编辑性 | layer count 是否合理；是否有语义命名；foreground/background/eyes/mouth 是否可单独编辑 |
| 稳定性 | 小变化输入是否导致 layer id 大幅变化 |
| 语义对齐 | SAM/SAM2 mask 与 SVG layer raster mask IoU |

### 5.5 人工评审

每次 benchmark 自动生成 HTML gallery：

```text
outputs/eval/YYYYMMDD-HHMM/
  metrics.csv
  per_sample.jsonl
  gallery.html
  animations/
  svgs/
  meshes/
  renders/
```

人工评审采用 pairwise：

- LivePortrait vs ToonCrafter interpolation
- ToonCrafter interpolation vs ToonCrafter driver
- TripoSR vs SF3D/TRELLIS/Hunyuan3D candidate
- current SVG vs VTracer vs layered vectorization

评分维度：

1. 身份保持
2. meme 冲击力
3. 时间一致性
4. artifact 少
5. 3D 可用性
6. SVG 可编辑性

汇总方式建议：win rate + Bradley-Terry/Elo，而不是简单平均分。

---

## 6. 优先级路线图

### P0：让当前系统可验证

1. **透传 ToonCrafter driver mode**
   - `FullPipelineRequest` 增加：
     - `use_toon_crafter_as_driver: bool`
     - `driver_num_frames: int`
   - `/animate`、`/generate`、`/batch` 增加 Form 参数和范围校验。
   - `FullPipeline.execute()` 创建 `AnimationRequest` 时透传。

2. **真实 ToonCrafter 适配**
   - 新增 `ToonCrafterBackend` capability check：
     - 权重是否存在；
     - 源码 import 路径是否存在；
     - 可调用 API 是 CLI、script、`lvdm` 还是封装 class；
     - 输入输出帧 shape、RGB/BGR、num_frames 语义。
   - 把当前假设的 `tooncrafter.inference.ToonCrafterInference` 改成可配置 adapter。

3. **验证 LivePortrait IP conditioning 是否真实生效**
   - 若官方 API 不支持 `ip_adapter_embeds`，不要继续在报告/API 中宣称它用于条件生成。
   - 可先把 IP embedding 用于评估、缓存、检索和一致性指标。

4. **建立 `scripts/eval_pipeline.py`**
   - 支持 `animation_only`、`svg_only`、`reconstruct_only`、`full`。
   - 输出 JSONL/CSV/gallery。
   - 首批指标先做轻量版：frame count、latency、CLIP/DINO similarity、LPIPS、SVG path/node/file size、mesh stats。

### P1：让输出质量有可比较 baseline

1. **动画 benchmark**
   - 对同一批 30 张图跑：
     - LivePortrait only
     - LivePortrait + ToonCrafter interpolation
     - LivePortrait peak + ToonCrafter driver
   - 产出 gallery 和人工 Elo。

2. **SVG baseline**
   - 集成 VTracer。
   - 对比：
     - current SAM+Bezier
     - OpenCV fallback
     - VTracer
   - 指标：raster LPIPS、node count、file size、layer editability。

3. **3D baseline**
   - 保留 TripoSR。
   - 先调研/适配 SF3D 作为第二 backend，因为它与 TripoSR 一样强调速度，但更接近生产资产。

4. **修正文档/代码漂移**
   - SAM VRAM 估算统一。
   - README、architecture、configs 中的 ToonCrafter driver 字段同步。
   - 把模型路径/第三方源码路径的实际安装方式写清楚。

### P2：引入前沿质量后端

1. **视频 quality backend**
   - 以统一接口接 Wan FLF2V / LTX multi-keyframe。
   - 不进入默认单进程 10GB 模式，而是作为独立 worker 或外部服务。

2. **3D quality backend**
   - TRELLIS/Hunyuan3D 做 quality mode。
   - 输出不仅是 OBJ/GLB，还包括材质、贴图、normal、渲染预览。

3. **语义 SVG**
   - 参考 Layered Image Vectorization：
     - 先做语义简化；
     - 再结构层级搭建；
     - 最后 raster-loss refinement。

4. **animated SVG / frame-consistent SVG**
   - 用 SAM2 做视频 mask propagation。
   - 让不同帧中的 eyes/mouth/body layer 保持同一 id。

### P3：产品化

1. 异步 job 队列，避免 FastAPI 长请求阻塞。
2. 按 source hash、expression、backend、resolution、seed 缓存中间结果。
3. 每个生成任务输出 `manifest.json`：

```json
{
  "source_hash": "...",
  "request": {},
  "model_versions": {},
  "backends": {},
  "metrics": {},
  "outputs": {},
  "warnings": []
}
```

4. Review UI：展示 A/B、多指标、人工 accept/reject。

---

## 7. 推荐的 6 周执行计划

### Week 1：工程真相对齐

- 暴露 ToonCrafter driver 参数。
- 修复文档/配置/API 漂移。
- 做 ToonCrafter/LivePortrait capability check。
- 增加真实模型 smoke test 开关：`EMOSVG_RUN_REAL_MODELS=1`。

### Week 2：评估 harness v0

- 实现 `scripts/eval_pipeline.py`。
- 支持 animation-only。
- 生成 `metrics.csv` 与 `gallery.html`。
- 指标：帧数、时长、latency、CLIP/DINO similarity、temporal diff、输出大小。

### Week 3：动画三路 benchmark

- 30 张 IP 图：
  - LivePortrait only
  - ToonCrafter interpolation
  - ToonCrafter driver
- 人工 pairwise review。
- 得出 driver mode 是否真正优于 interpolation 的证据。

### Week 4：SVG baseline

- 集成 VTracer。
- 与 SAM+Bezier 对比。
- 加 SVG rasterization 指标、node/path/file size 指标。

### Week 5：3D baseline

- 对 TripoSR 输出增加 multi-view render preview。
- 引入 mesh/GLB validator。
- 评估 SF3D 接入成本。

### Week 6：报告与路线收敛

- 生成第一版 leaderboard。
- 根据实测决定：
  - ToonCrafter 是否继续作为默认 driver；
  - 是否上 Wan/LTX 外部服务；
  - SVG 是否先走 VTracer 还是继续增强 SAM+Bezier；
  - 3D 是否优先接 SF3D。

---

## 8. 参考链接

### 身份与视觉特征

- CLIP, ICML 2021: https://icml.cc/virtual/2021/oral/9194
- OpenAI CLIP page: https://openai.com/research/clip
- ArcFace, CVPR 2019: https://huggingface.co/papers/1801.07698
- DINOv2: https://huggingface.co/papers/2304.07193
- IP-Adapter: https://huggingface.co/papers/2308.06721
- DreamBooth, CVPR 2023: https://cvpr.thecvf.com/virtual/2023/poster/23180
- PhotoMaker, CVPR 2024: https://cvpr.thecvf.com/virtual/2024/poster/30805
- InstantID: https://huggingface.co/papers/2401.07519

### 角色动画与图生视频

- First Order Motion Model, NeurIPS 2019: https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-a
- TPS Motion Model, CVPR 2022: https://huggingface.co/papers/2203.14367
- LivePortrait: https://liveportrait.github.io/
- LivePortrait paper: https://huggingface.co/papers/2407.03168
- Animate Anyone, CVPR 2024: https://cvpr.thecvf.com/virtual/2024/poster/31222
- MagicAnimate, CVPR 2024: https://cvpr.thecvf.com/virtual/2024/poster/29797
- MagicAnimate project: https://www.magicanimate.org/
- Champ: https://huggingface.co/papers/2403.14781
- AniPortrait: https://huggingface.co/papers/2403.17694
- EMO: https://huggingface.co/papers/2402.17485
- ToonCrafter, ACM TOG 2024: https://research.monash.edu/en/publications/tooncrafter-generative-cartoon-interpolation/
- ToonCrafter paper: https://huggingface.co/papers/2405.17933
- DynamiCrafter, ECCV 2024: https://eccv.ecva.net/virtual/2024/oral/1246
- SEINE, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html
- Stable Video Diffusion: https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets
- Wan2.1 GitHub: https://github.com/Wan-Video/Wan2.1
- Wan2.2 GitHub: https://github.com/Wan-Video/Wan2.2
- LTX-Video GitHub: https://github.com/Lightricks/LTX-Video
- LTX image-to-video docs: https://docs.ltx.video/open-source-model/usage-guides/image-to-video

### 单图 3D 与 3D 资产

- LRM, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/dcad3425f5c8c36b5b3885c091bf1257-Abstract-Conference.html
- Adobe LRM: https://research.adobe.com/publication/lrm-large-reconstruction-model-for-single-image-to-3d/
- TripoSR: https://huggingface.co/papers/2403.02151
- Wonder3D, CVPR 2024: https://cvpr.thecvf.com/virtual/2024/poster/29642
- MVDream, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/adbe936993aa7cf41e45054d8b72f183-Abstract-Conference.html
- SyncDreamer, ICLR 2024: https://iclr.cc/virtual/2024/poster/18828
- DreamGaussian, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/905202e21386913d8eac637c2b50f590-Abstract-Conference.html
- SF3D project: https://stable-fast-3d.github.io/
- SF3D Stability AI page: https://stability.ai/research/sf3d-stable-fast-3d-mesh-reconstruction-with-uv-unwrapping-and-illumination-disentanglement
- TRELLIS, CVPR 2025: https://www.microsoft.com/en-us/research/publication/structured-3d-latents-for-scalable-and-versatile-3d-generation/
- TRELLIS GitHub: https://github.com/microsoft/TRELLIS
- Hunyuan3D 2.0: https://huggingface.co/papers/2501.12202
- Hunyuan3D GitHub: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- MeshAnything: https://huggingface.co/papers/2406.10163

### SVG、分割与矢量化

- VTracer GitHub: https://github.com/visioncortex/vtracer
- VTracer docs: https://www.visioncortex.org/vtracer-docs
- SAM: https://huggingface.co/papers/2304.02643
- SAM2: https://huggingface.co/papers/2408.00714
- Meta SAM2 intro: https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/
- DiffVG, ACM TOG 2020: https://colab.ws/articles/10.1145/3414685.3417871
- DeepSVG, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/bcf9d6bd14a2095866ce8c950b702341-Abstract.html
- Im2Vec, CVPR 2021 Oral: https://geometry.cs.ucl.ac.uk/projects/2021/im2vec/
- VectorFusion, CVPR 2023: https://cvpr.thecvf.com/virtual/2023/poster/20940
- SVGDreamer, CVPR 2024: https://cvpr.thecvf.com/virtual/2024/poster/29501
- Layered Image Vectorization, CVPR 2025: https://cvpr.thecvf.com/virtual/2025/poster/34467
- Layered Image Vectorization project: https://szuviz.github.io/layered_vectorization/

### 评估 benchmark

- FVD: https://huggingface.co/papers/1812.01717
- LPIPS, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html
- RAFT, ECCV 2020: https://huggingface.co/papers/2003.12039
- VBench, CVPR 2024: https://cvpr.thecvf.com/virtual/2024/poster/30230
- EvalCrafter, CVPR 2024: https://cvpr.thecvf.com/virtual/2024/poster/29818
- EvalCrafter project: https://evalcrafter.github.io/
- VBench++: https://huggingface.co/papers/2411.13503
- DOVER, ICCV 2023: https://vqassessment.github.io/DOVER/
- GPTEval3D, CVPR 2024: https://gpteval3d.github.io/
- Eval3D, CVPR 2025: https://cvpr.thecvf.com/virtual/2025/poster/35203
