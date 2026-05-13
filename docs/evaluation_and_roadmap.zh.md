# emoSVG 评估与迭代路线图

## 1. 项目理解

emoSVG 是一个单进程、多模态资产生成管线：

1. 输入源角色图像
2. 提取 CLIP/IP 特征
3. 使用 LivePortrait 和可选 ToonCrafter 生成 Squash-and-Stretch 风格 meme 动画
4. 可选执行单图 3D 重建
5. 可选执行语义分层 SVG 矢量化

当前架构最强的定位不是“封装某个单点 SOTA 模型”，而是一个集成式、可控的 IP 角色资产工作流。它的竞争力应该从完整流程衡量：身份保持、卡通表现力、资产可编辑性，以及端到端资源成本。

## 2. 领域格局

| 方向 | 成熟基线 | 前沿方向 | 对 emoSVG 的启发 |
|---|---|---|---|
| 肖像与角色动画 | LivePortrait 使用隐式关键点进行高效、可控的肖像动画，相比许多纯扩散方案，在速度和可控性上更适合工程落地。来源: https://liveportrait.github.io/ | LTX-Video、Wan2.1 FLF2V、Wan2.2 I2V/Animate 等大型视频/图生视频模型有更强运动先验，但推理成本明显更高。来源: https://huggingface.co/Lightricks/LTX-Video, https://github.com/Wan-Video/Wan2.1, https://github.com/Wan-Video/Wan2.2 | 保留 LivePortrait 作为快速可控路径。大型视频模型更适合作为评测上限或可选高质量后端，而不是默认路径。 |
| 卡通插帧 | ToonCrafter 面向卡通插帧，解决传统线性插值或对应点方法在大幅非线性运动、遮挡/显露变化下容易失效的问题。来源: https://huggingface.co/papers/2405.17933 | 首末帧视频生成正在成为更通用的方案。Wan2.1 FLF2V 与 emoSVG 的 source/peak 边界帧设计尤其契合。来源: https://github.com/Wan-Video/Wan2.1 | 当前 driver mode 的设计方向正确，但必须先从端到端 API 暴露出来，并与 interpolation mode 做实测对比。 |
| 分割 | SAM 是稳健的图像分割基线；SAM 2 将能力扩展到图像和视频，引入流式记忆，并提升图像分割速度与效果。来源: https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/, https://huggingface.co/papers/2408.00714 | 视频感知的 mask 可用于稳定动画帧中的图层跟踪，也能提升 SVG/3D 提取的一致性。 | 当关注时间一致性时，SVG 分割应从静态 SAM ViT-H 迁移到 SAM 2 或提示式 mask stack。 |
| 单图 3D | TripoSR 快速实用，其论文报告在高端 GPU 上可实现亚秒级前馈单图 3D mesh 生成。来源: https://huggingface.co/papers/2403.02151 | Stable Fast 3D、TRELLIS、Hunyuan3D 2.x 等系统在带纹理/PBR 资产质量和表示灵活性上更强。来源: https://stability.ai/news/introducing-stable-fast-3d, https://microsoft.github.io/TRELLIS/, https://github.com/Tencent-Hunyuan/Hunyuan3D-2 | 保留 TripoSR 作为低延迟路径，同时增加后端抽象，为更高质量 mesh/PBR 路径预留接口。 |
| 栅格转 SVG | VTracer/Potrace 风格描摹仍是紧凑 SVG 的实用生产基线。来源: https://github.com/visioncortex/vtracer | Layered Image Vectorization via Semantic Simplification 和 DiffVG 风格优化说明，未来重点不只是轮廓追踪，而是语义化、紧凑、可编辑 SVG。来源: https://szuviz.github.io/layered_vectorization/, https://cseweb.ucsd.edu/~tzli/diffvg/ | 当前 SAM + Bezier 路径是合理原型。下一步应加入语义层级和基于 raster loss 的可微优化。 |
| 评估 | VBench/VBench++ 将视频质量拆成身份一致性、运动平滑度、闪烁、条件一致性等维度。来源: https://huggingface.co/papers/2311.17982, https://huggingface.co/papers/2411.13503 | 生成模型评测正在从单一分数转向诊断型指标和人类偏好对齐。 | emoSVG 应建立面向自身 IP-meme 工作流的任务评测集，而不是只依赖通用 FVD/FID。 |

## 3. 当前项目判断

### 优势

- 模块边界清晰，统一的 `ModelRegistry` 让重模型编排具备可测性和 VRAM 可控性。
- CPU fallback 路径让 CI 可以在没有模型权重的情况下验证接口契约。
- ToonCrafter 的两种模式概念上很强：interpolation mode 负责平滑，driver mode 负责从 source/peak 边界帧生成更夸张的卡通运动。
- 3D/SVG 输入可在原图和 peak keyframe 之间切换，这对产品很重要，因为创作者可能既需要标准姿态资产，也需要夸张表情资产。

### 风险与缺口

- `FullPipeline` 和 API 暴露了 `use_toon_crafter` 与 `frames_between`，但没有暴露 `use_toon_crafter_as_driver` 与 `driver_num_frames`。因此架构中最重要的动画路径目前无法从 `/generate` 或 `/animate` 触达。
- ToonCrafter 适配器假设存在打包良好的 `tooncrafter.inference.ToonCrafterInference` API。根据本地跑通记录，ToonCrafter 并不是标准 Python 包结构，因此真实 adapter 路径需要 real-weight 集成测试。
- 将 `ip_adapter_embeds` 传给 LivePortrait 是一个乐观扩展，需要对照真实 LivePortrait API 验证。否则身份条件可能被静默忽略。
- 当前测试主要是 fallback 或 mock 测试，验证了 shape、路由和文件产出，但没有验证真实视觉质量。
- 文档和代码存在漂移：SAM VRAM 估算不一致、IP fallback 注释不一致、driver-mode 请求字段没有进入 full pipeline。
- 3D/SVG 的 secondary input 依赖抽样后的 animation keyframe metadata。尤其在 smoothing 或 driver generation 后，它不一定总能选中真正的 peak frame。

## 4. 评估方案

### 数据集

构建一个小而诊断性强的 `eval_assets/` 评测集：

- 40 张扁平 IP 角色：吉祥物、贴纸、动漫风、emoji-like、非人类角色、透明 PNG、复杂背景 PNG。
- 20 张类人脸角色，用于 ArcFace/InsightFace 身份指标。
- 20 个带源 SVG 的合成/已知矢量资产，用于 SVG 重建和可编辑性评估。
- 20 个 Objaverse 或内部 3D 资产渲染成的 2D 图，保留已知 mesh ground truth，用于 3D 指标。
- 10 个对抗样例：极小眼睛、无嘴、多脸、透明孔洞、细线轮廓、文字/logo 区域。

每个样本应配套 manifest：

```json
{
  "id": "mascot_001",
  "category": "flat_mascot",
  "source": "path/to/image.png",
  "has_face": false,
  "has_svg_gt": true,
  "has_mesh_gt": false,
  "expected_primary_regions": ["head", "eyes", "mouth", "body"]
}
```

### 自动化指标

| 输出 | 指标 |
|---|---|
| 动画 | 首帧/末帧 CLIP 或 DINO 相似度、可选 ArcFace 相似度、前景 mask 时序 IoU、颜色漂移、temporal LPIPS/闪烁、光流平滑度、帧数/时长正确性、出框和 alpha 边界错误 |
| Meme 表现力 | 轮廓高宽变化、可检测 landmark/keypoint 位移、嘴部/眼部区域形变、shock/laugh/rage/cry/smug 预设之间的可分离性 |
| 3D | 多视角渲染与源图相似度、CLIP/DINO 图像对齐、顶点/面数、水密性、非退化面比例、法线一致性、自相交告警、GLB 校验；如有 GT，则计算 Chamfer distance 和 F-score |
| SVG | SVG rasterize 后与源图的 PSNR/SSIM/LPIPS、图层数、路径数、节点数、文件大小、渲染时间、语义 mask IoU、图层命名质量、可编辑性评分 |
| 系统 | p50/p95 延迟、峰值 VRAM、CPU RAM、模型加载时间、offload 次数、失败率、seed 可复现性、输出文件大小 |

### 人工评审

每周进行小规模 blind pairwise review：

- A/B: LivePortrait only vs ToonCrafter interpolation vs ToonCrafter driver。
- A/B: TripoSR vs 候选高质量 3D 后端。
- A/B: 当前 SVG vs VTracer vs semantic layered vectorization。

使用 1-5 分评价：

- 身份保持
- meme 冲击力
- 时间一致性
- 视觉 artifact
- 3D 可用性
- SVG 可编辑性

最终用胜率加 Bradley-Terry/Elo 聚合，让每轮迭代都有清晰 leaderboard。

### Benchmark Harness

新增类似 `scripts/eval_pipeline.py` 的评测脚本：

- 输入：manifest JSONL、backend config、output root
- 输出：per-sample JSON、aggregate CSV、HTML gallery
- 模式：`animation_only`、`svg_only`、`reconstruct_only`、`full`
- 对比基线：`fallback`、`live_portrait`、`live_portrait_toon_interp`、`live_portrait_toon_driver`

发布 gate 至少检查：

- endpoint 无回归
- failure rate 不上升
- peak VRAM 在配置预算内
- animation identity score 不超过设定容忍范围地回退
- 扁平图输入下 SVG 文件大小/节点数不爆炸

## 5. 迭代路线

### Phase 0: 让当前声明可验证

- 修复 architecture/README 漂移和编码问题。
- 将 `use_toon_crafter_as_driver` 与 `driver_num_frames` 暴露到 `FullPipelineRequest`、`/animate`、`/generate` 和 `/batch`。
- 增加 real-backend smoke test，通过显式环境变量开启，例如 `EMOSVG_RUN_REAL_MODELS=1`。
- 增加 backend capability checks，分别报告缺失 import、缺失权重、unsupported adapter API。
- 将 sampled-keyframe secondary input 替换为 animator 显式保存的 peak frame。

### Phase 1: 建立评估闭环

- 创建 eval manifest 和小型 seed dataset。
- 实现动画/SVG/3D/系统自动化指标。
- 每次评测生成 HTML 对比 gallery。
- 将 benchmark summary 存到 `outputs/eval/YYYYMMDD-HHMM/`。
- 定义 `"developer pass"`、`"visual pass"`、`"demo pass"` 三档发布阈值。

### Phase 2: 提升动画质量

- 用实测安全范围校准 `SquashParams -> LivePortrait exp_delta` 映射。
- 增加 mask-aware deformation，让非人脸吉祥物和扁平贴纸不要完全依赖人脸系数。
- 在相同 source/peak pairs 上比较 ToonCrafter driver、ToonCrafter interpolation、LTX-Video 和 Wan FLF2V。
- 增加可选参考动作模板：bounce、shake、pop、recoil、inhale、snapback。
- 将 identity drift 和 temporal flicker 作为一等指标持续追踪。

### Phase 3: 提升 SVG 与 3D 资产质量

- 增加 VTracer 作为 SVG 紧凑性生产基线。
- 升级或补充 SAM 2，以获得更好的分割和潜在的视频 mask propagation。
- 增加语义图层分组：background、body、eyes、mouth、accessories、outlines、highlights。
- 在 Bezier fitting 后加入 DiffVG-style 或 raster-loss refinement。
- 增加 3D 后端抽象：TripoSR 作为 fast mode，Hunyuan3D/TRELLIS/Stable Fast 3D 作为 quality modes。
- 增加 GLB 生产级 mesh cleanup 目标：尺度归一化、材质命名、texture atlas、manifold 检查。

### Phase 4: 产品化工作流

- 增加异步 job 管理，避免长请求阻塞。
- 按 source hash、expression、backend、resolution、seed 缓存 IP features 和模型输出。
- 增加轻量 review UI，用于查看 gallery、分数和人工 accept/reject。
- 每次生成输出 `manifest.json`，记录输入、seed、模型版本、指标、输出路径和 warning。

## 6. 优先级建议

下一轮 sprint 最建议做：

1. 端到端暴露 ToonCrafter driver mode；
2. 增加 eval harness 和 gallery；
3. 先跑一个 30 图 animation-only benchmark，对比 LivePortrait、ToonCrafter interpolation、ToonCrafter driver；
4. 等这些分数存在后，再决定是否替换或补充 ToonCrafter，引入更新的 I2V/FLF2V 模型。

这样可以让项目保持扎实：路线图可以有野心，但每次模型变化都必须在真实 IP-meme 工作流上打赢已有基线。
