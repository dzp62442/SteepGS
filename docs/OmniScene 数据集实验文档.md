# OmniScene 数据集实验文档（SteepGS / `comp_svfgs` 设计稿）

## 1. 文档状态与目标

本文档是实现前设计稿。当前阶段只确定数据契约、坐标系、预处理格式、SteepGS 接入方式、实验协议和验收标准，不创建 `comp_svfgs/` 实现代码，不修改训练流程，也不启动正式实验。文档审阅通过后再按本文实现。

目标是在 OmniScene 上以一个 bin 为一个独立场景运行 SteepGS：每个场景重新初始化一组高斯，只使用 6 张 context 图像优化，在同一条连续的优化轨迹上于 1k、5k、10k 迭代分别使用 18 张 target 图像渲染并评估，最后对 Center150 的 150 个场景做严格汇总。

正式对比的默认协议为：

- split：`center150`；
- 图像分辨率：`112x200`，可切换为 `224x400`；
- 训练视图：6；
- 评估视图：18，其中前 12 张是相邻时刻新视角，后 6 张是输入视角；
- 最大迭代数：10000；
- 评估里程碑：1000、5000、10000；
- 图像缩放参数：`-r 1`；
- 初始化：6 路 Metric3D 绝对尺度深度生成的彩色点云；
- 指标：PSNR、SSIM、LPIPS；
- 断点粒度：完整场景。样本内不保存 checkpoint，未完成场景从 0 重跑。

以下内容不在本轮范围内：生成或修改 Center150、加载动态物体掩码、加载 DepthAnything 相对深度、计算 PCC、使用 target 图像参与优化、正式运行 150 场景实验。

## 2. 参考实现与已确认事实

### 2.1 参考来源

本方案主要对照以下实现：

- `../depthsplat/docs/OmniScene数据集实验文档.md`；
- `../depthsplat/src/dataset/dataset_omniscene.py`；
- `../depthsplat/src/dataset/utils_omniscene.py`；
- `../SVF-GS/data/omniscene_dataset.py`；
- `../SVF-GS/data/transforms/loading.py`；
- `../SVF-GS/scripts/generate_omniscene_center150.py`，仅用于理解并校验既有清单，不复制生成逻辑；
- `../DropGaussian_release/comp_svfgs/dataset_omniscene.py` 与 `../DropGaussian_release/scripts/run_omniscene.py`；
- `../Octree-GS/comp_svfgs/dataset_omniscene.py`、`comp_svfgs/omniscene_preprocess.py`、`run_omniscene.py` 及其测试。

用户所指的 DepthSplat 文档在工作区中的实际文件名不含空格。当前 DepthSplat 代码只直接实现了 `train/val/test` 三种 stage；文档提到的 `demo`、以及本任务要求的 `center150`，不能从该类直接照搬。SVF-GS 当前版本已将用户提到的 `data/dataloader.py` 重命名为 `data/omniscene_dataset.py`，因此本次实际审阅的是重命名后的数据类和 `data/transforms/loading.py`。SteepGS 的 Center150 语义将以 SVF-GS 已存在的清单为唯一来源，并参考已完成的优化式基线做只读校验。

### 2.2 对本机 Center150 的只读核验

对现有 `bins_center150_v1.json` 和相应数据做过全量元数据检查，结果如下：

- 150 个 bin token，全部唯一；
- 覆盖 150 个不同的官方 val 场景；
- 每个 bin 的 6 路相机都至少包含索引 0、1、2；
- 6 个 context 视图所需的 RGB、内参 JSON、Metric3D depth/conf 文件全部存在；
- 150 个样本共检查 2700 个 context/target 视图，位姿旋转矩阵正交；
- 224x400 数据的主点均为 `(cx, cy)=(200,112)`，缩放到 112x200 后为 `(100,56)`；
- 焦距不是全局常数：检查到的 `fx` 约为 209.91～329.05，`fy` 约为 211.00～353.55，因此必须逐视图保留焦距；
- 在 `finite && depth>0 && confidence>0.3` 下，输入深度范围约为 0.459～169.25 米，其中约 3.86% 的有效像素深度大于 100 米。

这些检查只证明当前数据布局满足设计前提，不等同于 SteepGS 已经能够正确训练；坐标投影和真实渲染仍需按第 12 节验收。

## 3. DepthSplat 与 SteepGS 的流程差异

| 项目 | DepthSplat 前馈流程 | SteepGS 逐场景优化流程 |
| --- | --- | --- |
| 模型状态 | 多场景共享网络权重，一次 forward 预测当前样本高斯 | 每个 bin 新建独立 GaussianModel，反复优化后丢弃/保存该场景状态 |
| context | 作为网络输入 | 6 张图像进入随机视角训练采样和重建损失 |
| target | 可参与前馈训练监督与统一测试 | 绝不参与优化，只在里程碑渲染和评估 |
| 深度 | 模型可自行预测；数据加载按任务可带相对深度 | Metric3D 绝对深度只用于生成初始化 PLY，不作为训练监督 |
| 数据入口 | PyTorch Dataset → DataModule → ModelWrapper | 先生成 SteepGS 可识别的逐场景磁盘结构，再由 Scene/Camera 加载 |
| 迭代含义 | 对共享模型做跨样本训练 step | 对一个场景的高斯做 0→10k 连续优化，完成后处理下一个场景 |
| checkpoint | 面向共享网络训练 | Center150 不保存样本内 checkpoint；恢复粒度是完整场景 |
| 统计 | 通常按 batch/test 输出汇总 | 每个场景三组里程碑结果，150 场景全部完整后做宏平均 |

因此本项目不能只注册一个 Dataset 类。还需要预处理层、SteepGS 专用 Scene reader、训练中的里程碑评估钩子，以及负责跨场景状态管理和汇总的启动器。

## 4. OmniScene 数据契约

### 4.1 数据根目录

默认数据根目录按以下顺序解析：

1. 命令行 `--data-root`；
2. 环境变量 `OMNISCENE_ROOT`；
3. 工作区默认 `../SVF-GS/data/nuScenes`。

第三项在当前机器上最终指向现有 OmniScene 数据，不在 SteepGS 内复制整套原始数据。解析后必须使用真实绝对路径写入协议记录，路径不存在时立即报错。

数据版本固定为 `interp_12Hz_trainval`。所有模式都读取 `bin_infos_3.2m/<bin_token>.pkl`。

### 4.2 模式定义

| mode | token 来源 | 用途 |
| --- | --- | --- |
| `center150` | `bins_center150_v1.json` | 默认、正式对比；保持原顺序，只读且严格校验 |
| `train` | `bins_train_3.2m.json` 全量 | 接口兼容与调试，不是当前正式协议 |
| `val` | `bins_val_3.2m.json[:30000:3000][:10]` | 10 个样本快速验证 |
| `test` | `bins_val_3.2m.json[0::14][:2048]` | 与现有 mini-test 语义兼容 |
| `demo` | 参考代码中的固定 `bins_demo` | 少量指定样本调试 |

Center150 加载器不得包含“清单不存在时自动生成”的分支，也不得修复、重排或覆盖清单。加载前至少验证：

- JSON 只有可用的 `bins` 列表；
- 恰好 150 个合法且唯一的 bin token；
- token 符合 `^scene[0-9a-f]+_bin[0-9]+$`，不含路径分隔符；
- 解析出的 scene token 恰好 150 个且均唯一；
- 每个 `bin_infos_3.2m/*.pkl` 存在；
- 顺序原样保留，并把清单内容 SHA-256 写入实验协议。

### 4.3 相机顺序与视图身份

6 路相机顺序固定为：

1. `CAM_FRONT`；
2. `CAM_FRONT_RIGHT`；
3. `CAM_FRONT_LEFT`；
4. `CAM_BACK`；
5. `CAM_BACK_LEFT`；
6. `CAM_BACK_RIGHT`。

每个 bin 的 context 恰好为上述 6 路相机的 `sensor_info[camera][0]`。

target 恰好为 18 张，顺序固定为：先按上述相机顺序依次加入每路的索引 1、索引 2，共 12 张；再按相机顺序加入 6 张 context。每个里程碑都使用同一组 18 张评估，不是三个里程碑各分配 6 张。

每个视图保存稳定的 `view_id`，包含 target 序号、相机名、帧索引和角色，例如：

```text
target_00_CAM_FRONT_t1
target_01_CAM_FRONT_t2
...
target_11_CAM_BACK_RIGHT_t2
target_12_CAM_FRONT_context
...
target_17_CAM_BACK_RIGHT_context
```

加载时验证 6 个 context ID 和 18 个 target ID 均唯一。target 中的最后 6 张与 context 是同一数据身份，预处理阶段复用对应 PNG 文件，不做第二次有损编码。

### 4.4 返回内容

`comp_svfgs/dataset_omniscene.py` 对单个 bin 返回：

```text
scene/bin_token
context:
  image        [6, 3, H, W], float32, [0, 1]
  intrinsics   [6, 3, 3], 像素单位
  c2w          [6, 4, 4], OpenCV camera → key-frame LiDAR/world
  depth_metric [6, H, W], float32, 米制 z-depth
  confidence   [6, H, W], float32
target:
  image        [18, 3, H, W], float32, [0, 1]
  intrinsics   [18, 3, 3], 像素单位
  c2w          [18, 4, 4], 同一 OpenCV/世界约定
  view_id      18 个稳定字符串
```

动态物体 mask、相对深度、射线、PCC 所需字段均不加载。target depth/conf 也不加载，因为它们既不参与初始化，也不参与当前指标。

## 5. 路径转换、缩放与图像处理

`load_conditions` 保留 DepthSplat/SVF-GS 已使用的路径转换规则：

```text
原始 data_path
  samples → samples_param_small
  sweeps  → sweeps_param_small
  .jpg    → .json                 # 相机内参

原始 data_path
  samples → samples_small
  sweeps  → sweeps_small          # RGB

small RGB path
  samples_small → samples_dptm_small
  sweeps_small  → sweeps_dptm_small
  .jpg          → _dpt.npy / _conf.npy
```

实现时保留转换结果，但会补充防御性检查：每次替换必须命中预期的数据目录段，最终路径必须仍位于解析后的 OmniScene 根目录中，文件必须存在。这样既保持现有数据约定，又避免普通字符串替换静默命中错误片段。

RGB 统一转为三通道。预处理输出使用 PNG，避免从源 JPEG resize 后再次写成有损 JPEG。若源尺寸不同于目标尺寸，按下式同步缩放内参：

```text
scale_w = target_w / source_w
scale_h = target_h / source_h
fx' = fx * scale_w,  cx' = cx * scale_w
fy' = fy * scale_h,  cy' = cy * scale_h
```

RGB、depth 和 confidence 采用与参考加载器一致的双线性 resize；有效掩码必须在 resize 完成后重新计算，不能先阈值化再插值。正式有效条件严格使用：

```text
isfinite(depth) && depth > 0 && isfinite(confidence) && confidence > 0.3
```

Metric3D 文件是米制相机 z-depth，不乘 1000、不除 1000、不做每场景尺度归一化。224x400 与 112x200 只改变采样密度和 K，不改变深度值的物理单位。

## 6. 坐标系、相机接入与初始化点云

### 6.1 唯一坐标链路

本适配统一使用关键帧 LiDAR 坐标作为 world：

```text
OpenCV camera: x 向右，y 向下，z 向前
sensor2lidar_transform: OpenCV camera → key-frame LiDAR/world
```

预处理 JSON 保存原始 OpenCV `c2w`。SteepGS 专用 reader 直接执行：

```text
w2c = inverse(c2w)
R = transpose(w2c[0:3, 0:3])
T = w2c[0:3, 3]
```

这里不调用 Blender reader，也不做 `diag(1,-1,-1,1)` 的 Y/Z 翻转。DropGaussian 采用“写入 OpenGL c2w、reader 再翻回 OpenCV”，Octree-GS 采用 `no_flip_yz=true`；两者在正确使用时与本方案代数等价，但 SteepGS 使用独立 OpenCV reader 可以避免双重翻转和格式歧义。

SteepGS 会计算 NeRF++ normalization，但当前 `loadCam()` 实际没有把 `translate/scale` 传给 `Camera`，训练中只消费 `radius` 作为位置学习率和致密化相关的尺度。因此 OmniScene reader 明确保持 world transform 为 identity，不对相机或 PLY 额外平移/缩放；radius 仍由 6 个训练相机中心按项目原逻辑计算并记录，不在数据适配阶段调参。

### 6.2 逐视图内参

预处理的每个 frame 都写入自己的 `fl_x/fl_y/cx/cy/width/height`。reader 按视图计算：

```text
FovX = 2 * atan(width  / (2 * fx))
FovY = 2 * atan(height / (2 * fy))
```

不能像原 Blender reader 一样共享顶层 `camera_angle_x`。当前 SteepGS 投影矩阵是对称视锥，不能表达偏心主点；已检查的 Center150 主点精确位于图像中心，因此逐视图 `fx/fy` 的 FOV 表达对当前协议是准确的。loader 仍保存并验证 `cx/cy`，若偏离 `(width/2,height/2)` 超过 `1e-4` 像素则 fail-fast，而不是静默忽略。

这使实现无需修改 `scene/cameras.py`、`utils/camera_utils.py` 或 `utils/graphics_utils.py`，并保持原 Colmap/Blender 路径不变。若未来数据出现偏心主点，再单独引入并验证非对称投影矩阵。

### 6.3 深度反投影和 PLY

仅使用 6 个 context 视图生成初始化点云。对有效像素 `(u,v)` 和米制 z-depth `d`：

```text
x = (u - cx) * d / fx
y = (v - cy) * d / fy
z = d
p_world = c2w_opencv @ [x, y, z, 1]^T
```

颜色取同一个 resize 后 RGB 像素，PLY 以 float32 XYZ、零法线和 uint8 RGB 写入。不得对相机点的 y/z 做翻转，不从 target 生成点，也不在深度缺失时回退到随机点云；缺 depth/conf、有效点为空、点或颜色出现非有限值时，该场景预处理直接失败。

按用户给定的有效条件不额外截断大于 100 米的深度。SteepGS 当前 CUDA rasterizer 的 frustum 检查只剔除相机 z 小于等于 0.2 的点，没有按 `zfar=100` 做远平面剔除；因此本方案不为此改动基础 Camera 类，但会把大于 100 米点的正深度和投影检查作为实现验收项。

## 7. 预处理目录规划

建议目录为：

```text
output/
├── omniscene_preprocessed/
│   └── center150_112x200/
│       ├── protocol.json
│       ├── 001_<bin_token>/
│       │   ├── images/
│       │   │   ├── context_00_CAM_FRONT.png
│       │   │   ├── ...
│       │   │   ├── novel_00_CAM_FRONT_t1.png
│       │   │   └── ...
│       │   ├── transforms_train.json
│       │   ├── transforms_test.json
│       │   ├── points3D.ply
│       │   └── manifest.json
│       └── 150_<bin_token>/
└── omniscene_results/
    └── center150_112x200/
        ├── protocol.json
        ├── 001_<bin_token>/
        ├── ...
        ├── 150_<bin_token>/
        ├── center150_metrics_summary.json
        └── center150_metrics_summary.txt
```

需求中的 `01_<bin_token>` 解释为“有序编号前缀 + bin token”。因为正式集合有 150 个样本，为保证文件系统字典序与清单顺序一致，实际采用固定三位 `001_`～`150_`；第一项是 `001_<bin_token>`。如果 `01_` 被要求为所有目录共同的字面前缀，需要在实现前另行确认，因为那样不能提供序号排序信息。

`transforms_train.json` 引用 6 张 context；`transforms_test.json` 按第 4.3 节顺序引用 12 张 novel 和同一批 6 张 context，共 18 个 frame。每个 frame 保存 `view_id`、相对图像路径、原始 OpenCV c2w、逐视图 K 和角色。

`manifest.json` 至少记录：

- 预处理格式版本；
- bin token、split 序号、分辨率、相机顺序、置信度阈值；
- 6/18 个 view ID 及其原始 RGB、K、pose、depth/conf 身份；
- 原始文件真实路径、尺寸及内容摘要；
- K/c2w 数值摘要、有效点数量、PLY 点数；
- transforms、PNG、PLY 的文件列表、尺寸和摘要；
- 相关适配代码版本。

预处理先写到同一父目录下的场景专用临时目录；全部文件可读取、数量正确且 manifest 自洽后，才原子发布为稳定场景目录。缓存命中必须重新核对 manifest 与当前 split、源文件和参数指纹；任一不一致都重新预处理，不能仅凭目录存在跳过。快速路径先比较真实路径、文件大小和高精度 mtime；元数据变化时才重新计算内容 SHA-256。这样完整场景无需解码 RGB/depth，也不会放弃对源数据变化的检测。

## 8. 代码组织与最小集成面

文档通过后计划新增：

```text
comp_svfgs/
├── __init__.py
├── dataset_omniscene.py       # split、bin、RGB/K/c2w/depth/conf 加载与校验
├── omniscene_preprocess.py    # 场景目录、transforms、PLY、manifest
└── omniscene_evaluation.py    # 18 视图渲染、指标、PNG/JSON 原子保存
scripts/
└── run_omniscene.py           # 单场景编排、协议、跳过/重跑和最终汇总
tests/
└── test_omniscene.py          # 无 GPU 协议测试及可选 GPU smoke
```

计划对现有代码只做以下接入：

- `scene/dataset_readers.py`：增加独立 OmniScene reader，读取 raw OpenCV c2w 和逐帧焦距；
- `scene/__init__.py`：优先依据 OmniScene manifest 选择上述 reader，普通 Colmap/Blender 判定保持原样；
- `train.py`：增加受显式参数控制的严格里程碑评估与纯训练计时；正式路径不再依赖宽松的 `metrics.py`；
- `arguments/__init__.py`：评审通过后可将明显拼写错误 `inv_covar` 修正为渲染器支持的 `inv_cov`；无论是否修正，正式启动器都显式传入 `inv_cov`。

原则上不修改 `scene/cameras.py`、`utils/camera_utils.py`、`utils/graphics_utils.py`、`scene/gaussian_model.py`、`render.py` 和 `metrics.py`。README 中示例的 `--densitf_strategy` 是拼写错误，可作为一个独立、受限的一行文档修正，不作为 OmniScene 正确运行的隐式前提。

SteepGS 当前没有 `n_views` 参数。本适配不新增“截取前 N 张”的模糊接口：正式数据 manifest 和 reader 强制训练视图恰好为 6，训练采样自然覆盖全部 6 张。启动器的协议记录写入 `train_view_count=6`，若不是 6 则硬失败。这在语义上满足 `n_views=6`，同时避免额外参数造成无意丢帧。

## 9. 单场景主流程与命令行

### 9.1 “单阶段”的定义

对每个 bin，启动器按以下顺序执行：

1. 只读取 Center150 token 和已有 manifest/结果元数据，先判断是否完整；完整场景在读取原始 RGB/depth 前快速跳过；
2. 未完成场景检查预处理缓存；缓存有效则复用，否则加载该 bin 并预处理；
3. 为该场景创建隔离的工作结果目录，启动一个 SteepGS 训练进程；
4. 在同一个模型、同一个优化器和同一条 0→10k 连续轨迹中，于 1k/5k/10k 暂停训练做旁路评估；
5. 10k 后保存最终 PLY，严格校验全部产物，最后原子写 completion marker 并发布稳定结果目录；
6. 继续下一个 bin。全部 150 个稳定结果通过检查后才生成汇总。

这里的“单阶段”不是先把全数据预处理完再统一训练，也不是分别启动 1k、5k、10k 三次独立优化。三个里程碑必须来自同一条训练轨迹。

### 9.2 默认命令

项目根目录下的默认命令保持为：

```bash
python scripts/run_omniscene.py
```

它等价于以下正式配置：

```text
--mode center150
--resolution 112x200
--iterations 10000
--eval-iterations 1000 5000 10000
--confidence-threshold 0.3
--seed 0
--gpu 0
```

可覆写示例：

```bash
python scripts/run_omniscene.py --resolution 224x400

python scripts/run_omniscene.py \
  --mode val \
  --iterations 1000 \
  --eval-iterations 1000
```

启动器使用当前解释器 `sys.executable` 调用 `train.py`，避免意外使用系统 Python/pip。GPU 通过子进程启动前设置 `CUDA_VISIBLE_DEVICES`，不能等 torch 导入后再切换。

正式训练命令由启动器构造，关键参数固定包含：

```text
--eval -r 1 --no_gui
--iterations 10000
--test_iterations 1000 5000 10000
--save_iterations 10000
--densify_strategy steepest
--S_estimator inv_cov
```

`steepest` 是运行 SteepGS 方法而不是回退到原始 3DGS ADC 的显式选择；`inv_cov` 也避免当前默认值 `inv_covar` 与 renderer 查找表不一致。若构造出的正式命令缺少二者，启动器必须拒绝运行。

模型专用额外参数可通过 `--extra-train-args` 透传，但 `source_path/model_path/resolution/iterations/test_iterations/save_iterations/checkpoint/start_checkpoint/densify_strategy/S_estimator` 等协议关键参数禁止从透传区覆盖。任何合法覆写都会进入协议指纹，并使用独立结果目录，不能与默认结果混用。

## 10. 里程碑评估、计时与统计

### 10.1 保持 SteepGS 原生迭代语义

当前 `train.py` 在 `optimizer.step()` 之前调用 `training_report()` 和保存逻辑。因此标记为 iteration `k` 的评估状态对应完成了第 k 次前向/反向，但只完成了 k-1 次 optimizer update；最终 10k PLY 也沿用这一原生语义。

本适配不偷偷重排 optimizer、致密化或保存顺序，以免改变原方法。文档和结果协议会记录这一点，并用短跑验证评估开关不会改变训练相机采样序列、Gaussian 数量或最终模型。

### 10.2 指标口径

每个里程碑用当前内存中的 GaussianModel 渲染 18 张 target，渲染值和 GT 都 clamp 到 `[0,1]`，不使用动态 mask，不裁除图像区域。使用本项目已有实现：

- PSNR：`utils.image_utils.psnr`；
- SSIM：`utils.loss_utils.ssim`；
- LPIPS：`lpipsPyTorch.modules.lpips.LPIPS(net_type="vgg")`。

LPIPS 模型在单场景训练进程中只实例化一次并设为 eval，不按视图重复创建。每个视图保存 PSNR/SSIM/LPIPS；场景主指标为 18 张图的算术平均。另保存前 12 张 novel-only 平均作为诊断项，但正式 Center150 主汇总仍以用户指定的 all-18 指标为准。

150 场景汇总采用等场景权重的宏平均，并报告场景间标准差。因为每场景都必须有 18 张视图，all-18 的宏平均数值上也等价于把 2700 张图等权平均；逐场景和逐视图结果仍完整保留，便于追溯。

### 10.3 纯训练耗时

`training_time_seconds` 定义为训练循环累计 wall time，包含：训练相机选择、学习率更新、前向渲染、loss、backward、optimizer、致密化与剪枝。它不包含：

- 数据预处理和进程启动；
- 模型/相机初始化；
- 里程碑的 18 视图渲染和 PSNR/SSIM/LPIPS；
- PNG、JSON、日志、PLY 和 completion 文件 I/O。

在开始计时、暂停评估和恢复计时的边界调用 `torch.cuda.synchronize()`，再使用 `time.perf_counter()` 累加。1k、5k、10k 记录的都是从本场景第 1 次迭代开始的累计值，必须非负且单调。该字段只能称为“累计纯训练时间”，不能称为端到端耗时。

旁路评估前后保存并恢复 Python `random`、NumPy、Torch CPU、Torch CUDA RNG 状态；正式协议固定记录 seed 和确定性设置。CUDA 仍可能存在非确定性，因此还需按第 12 节做有/无里程碑评估的一致性实验。

### 10.4 单场景结果

```text
<result_root>/<scene_name>/
├── cfg_args
├── train_log.txt
├── protocol.json
├── metrics_1000.txt
├── metrics_5000.txt
├── metrics_10000.txt
├── metrics_1000.json
├── metrics_5000.json
├── metrics_10000.json
├── training_time_1000.txt
├── training_time_5000.txt
├── training_time_10000.txt
├── test/
│   ├── ours_1000/{renders,gt}/     # 各18张
│   ├── ours_5000/{renders,gt}/     # 各18张
│   └── ours_10000/{renders,gt}/    # 各18张
├── point_cloud/iteration_10000/point_cloud.ply
└── center150_complete.json
```

不创建 `chkpnt*.pth`，也不保存 1k/5k PLY。1k/5k 状态只在同一次训练进程中就地渲染和评估。

JSON 是严格完成检查和后续分析的规范来源，保存逐视图、all-18、novel-12 及协议摘要；同名 TXT 只保存 all-18 的 PSNR/SSIM/LPIPS，便于人工查看并与其它优化式基线的文件命名保持一致。

## 11. 完成态、重跑与协议指纹

### 11.1 完成条件

一个 Center150 场景只有同时满足以下条件才允许 skip：

- 预处理 manifest 与当前源文件、split、K/pose、分辨率、阈值和格式版本指纹一致；
- 结果 protocol 中影响实验语义的参数、seed、split 内容及顺序与当前运行一致；代码、Git 和软件环境信息仅用于溯源，不阻止续跑；
- 1k、5k、10k 各有恰好 18 张 render 和 18 张 GT，名称与 view ID 一致，文件可解码、尺寸正确且非空；
- 三个里程碑的 JSON/TXT 一致，逐视图及平均 PSNR/SSIM/LPIPS 完整且均为有限值；
- 三个累计训练时间非负且单调；
- 10k PLY 存在、可读取、点数大于 0，XYZ/RGB 有限；
- `center150_complete.json` 最后原子写入，且其中的协议/产物摘要与当前文件一致。

skip 时不重写 completion marker。任何缺件、NaN、不可解码图片、数据或实验语义不匹配、目录中混入不匹配产物，都判定为未完成，并打印明确拒绝原因。

### 11.2 不完整场景

不完整场景不恢复 checkpoint。启动器先用 `realpath` 确认清理目标严格位于当前实验结果根目录、且只对应当前 scene name；随后清理或替换该样本的临时/不完整结果，从 0 重新训练。预处理缓存保留，场景失败日志保存在其结果目录之外，避免随重跑清理掉诊断信息。

稳定结果目录只接收已经完整校验的工作目录，防止新旧里程碑产物混合。若正式运行发现任何不完整样本，已有全局 summary 先标记失效；只有再次验证全部 150 场景后才原子重建 summary。

### 11.3 协议指纹

协议文件同时保存实验配置与运行溯源信息。完整 SHA-256 仍会记录，但不直接作为断点续跑的准入条件。至少记录：

- Center150 JSON 内容、顺序和数据版本；
- 每场 6/18 个 view ID、RGB/K/c2w 及初始化 depth/conf 身份；
- 分辨率、置信度阈值、迭代数、里程碑、seed；
- 所有生效的模型/优化/渲染参数；
- SteepGS commit、dirty 状态、相关适配文件摘要和 submodule revision，均仅用于溯源；
- Python、PyTorch、CUDA、LPIPS backbone/权重版本，均仅用于溯源。

断点续跑只比较会改变实验含义的稳定字段：split 内容、样本顺序、分辨率、置信度阈值、迭代/评估里程碑、seed、训练参数、指标协议，以及场景 token/view/预处理数据身份。Git commit、dirty 状态、代码文件哈希、软件版本和由这些信息引起的完整指纹变化不会阻止复用。这样在实验运行期间提交相同代码，进程若随后中断，新代码仍能跳过完整场景，并将 `.work` 对应的未完成场景从 0 重跑。真正改变上述实验语义时仍拒绝混用结果，并要求使用新的结果目录。

## 12. 实现后的验收计划

在启动 Center150 正式实验前依次通过以下门槛。

### 12.1 无 GPU 单元测试

- 各 mode 的 token 来源和顺序正确；Center150 缺失、重复 scene、非法 token、缺 pkl 时硬失败；
- context 恰好 6，target 恰好且唯一 18，顺序为 12 novel + 6 context；
- 路径转换只命中预期目录，112x200/224x400 的 RGB/depth/conf/K 尺寸一致；
- `c2w` 最后一行为 `[0,0,0,1]`，旋转正交误差小于 `1e-5` 且行列式为正；
- resize 后主点离图像中心不超过 `1e-4` 像素；
- 反投影点在源相机中 z 为正，投回原像素的最大误差小于 `1e-4` 像素；
- PLY 读回点数、XYZ 和 RGB 与写入前一致，数值有限；
- 预处理缓存指纹、临时目录发布、实验语义冲突、安全清理、提交后兼容续跑、完整 skip、不完整重跑和 150 场景汇总均有测试；
- 构造出的命令含 `-r 1`、`steepest`、`inv_cov`，不含 checkpoint 参数，并拒绝关键参数从透传区覆盖。

### 12.2 坐标和真实数据检查

- 选一个 Center150 bin，将初始化 PLY 重投影回 6 张源图，抽样点误差小于 0.5 像素，颜色取样位置一致；
- 与 DropGaussian/Octree-GS 对同一 bin、同一分辨率和阈值生成的 world points 做数量、包围盒和抽样坐标对照；
- 检查每个相机前方点占比、相机中心和相机朝向，防止双重 Y/Z flip；
- 专门抽取深度大于 100 米的有效点，确认进入 SteepGS reader 后仍为正 z，投影误差小于 0.5 像素且未被 rasterizer 错误剔除；
- 未训练初始化渲染必须方向正确、非空；不以“进程没有报错”代替几何正确性。

### 12.3 回归与 GPU smoke

- 原 Colmap 和 Blender reader 至少各做一次加载回归，确认专用 OmniScene 分支没有改变原语义；
- 用一个场景运行短流程，覆盖首次 densification（默认 500 iteration 之后）、一次里程碑评估、最终 PLY 和完成判定；
- 固定 seed 分别运行“开启里程碑评估”和“关闭旁路评估”的等长短实验，训练相机序列与 Gaussian 数量必须一致；在确定性允许范围内最终参数最大绝对差不超过 `1e-6`，否则定位并消除评估副作用；
- 人工删除一个 metrics、一个 render 和 completion marker，分别验证不会误 skip；
- 验证一个完成样本能在加载 RGB/depth 前快速跳过，一个不完整样本保留预处理并从 0 重跑；
- 先运行 `val` 或单个 Center150 样本并人工检查结果，再由用户决定是否启动完整 150 场景实验。

## 13. 专家评审后的最终取舍

本设计经过三份独立首轮评审（几何/坐标、实验协议、SteepGS 集成）和两份交叉复审。收敛后的关键取舍是：

- 采用独立 OpenCV reader，不借 Blender 双重转换；
- 逐视图保留焦距，当前居中主点走现有对称投影，偏心数据拒绝加载；
- 绝对深度只生成 PLY，严格 `confidence>0.3`，不使用随机点云兜底；
- 保留 SteepGS 原生 optimizer/致密化顺序，里程碑评估不改变训练轨迹；
- 不新增 `n_views` 截断接口，以 6 视图强校验表达协议；
- 形式化预处理缓存、实验参数和完成态，不以单个 marker 或进程退出码代替完成；
- Center150 无样本内 checkpoint，不完整样本从 0 重跑；
- 正式汇总以 all-18、150 场景等权宏平均为主，同时保留逐视图、novel-only 和离散度供审计。

文档审阅通过后，开发按“loader/预处理 → Scene 接入 → 里程碑评估 → 启动器与恢复 → 单元测试和单场景几何验收”的顺序实施。在上述验收完成前不启动正式 Center150。
