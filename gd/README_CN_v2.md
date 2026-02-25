# GreenDiff 中文总文档（基于当前代码实现，v2）

## 源码基线
本文件基于当前仓库代码实现整理，重点覆盖以下模块与脚本（按功能分组）：

- 核心模型
  - `gf/models/vae.py:24`
  - `gf/models/latent_green.py:64`
  - `gf/models/diffusion.py:106`
  - `gf/models/condition_encoder.py:5`
  - `gf/models/student.py:23`
- 推理与引导
  - `gf/inference/teacher_sampler.py:13`
  - `gf/guidance/latent_guidance.py:5`
  - `gf/guidance/restart.py:7`
- 数据与物理前向
  - `gf/data/dataset.py:19`
  - `gf/data/kpm_forward.py:6`
- 训练/测试脚本
  - `gf/train/train_vae.py:22`
  - `gf/train/train_latent_green.py:25`
  - `gf/train/train_diffusion.py:31`
  - `gf/train/train_student.py:24`
  - `gf/test_vae.py:39`
  - `gf/test_green.py:79`
  - `gf/test_latent_green.py:60`
  - `gf/test_diffusion.py:61`
- 工具与配置
  - `gf/utils/ldos_transform.py:68`
  - `gf/utils/loss_align.py:7`
  - `gf/configs/default.yaml:1`
  - `gf/run_pipeline.py:71`

## 说明
- 本文档按“当前代码实现”组织，不按论文理想化公式或历史版本行为组织。
- 本版已同步近期修复的若干关键行为（如 diffusion 稀疏采样端点、VAE decoder 配置生效、`test_diffusion.py` 的 `phys_mse` 比较空间一致性等）。
- 旧 checkpoint 可能与当前实现存在行为差异（尤其采样与评估路径）；文档中会在对应章节标注。
- `gf/configs/default.yaml` 的中文注释在部分终端中可能乱码，但字段名与数值本身可正常使用。

## 1. 项目概览（主链路）
### 1.1 目标（代码实现视角）
GreenDiff 当前实现的主目标是：在给定观测 LDOS（`g_obs`）条件下，生成势场 `V`，并使用潜空间物理代理（`LatentGreen`）在训练或推理中提供物理一致性约束。

从代码调用关系看，核心链路可概括为：

`g_obs -> ConditionEncoder -> LatentDiffusion -> z -> VAE.decode -> V`

并配有一条教师物理代理链路：

`z -> LatentGreen -> g_pred (surrogate LDOS)`

### 1.2 模块关系（当前实现）
- 数据生成侧：`GFDataset` 调用 `KPMForward` 生成/读取 `g_obs`，定义于 `gf/data/dataset.py:19`、`gf/data/kpm_forward.py:6`
- 潜空间建模：`VAE` 将 `V <-> z` 映射，定义于 `gf/models/vae.py:24`
- 条件扩散：`LatentDiffusion` 在潜空间做条件生成，定义于 `gf/models/diffusion.py:106`
- 物理代理：`LatentGreen` 近似 `z -> LDOS`，定义于 `gf/models/latent_green.py:64`
- 推理封装：`TeacherSampler` 串起 diffusion / VAE / guidance / restart，定义于 `gf/inference/teacher_sampler.py:13`

### 1.3 推荐阅读顺序（新用户）
1. 先看“统一张量与空间约定”（第 2 章）
2. 再看 `VAE`（第 4 章）和 `LatentDiffusion`（第 7 章）
3. 再看 `TeacherSampler`（第 8 章）和训练脚本（第 9 章）
4. 最后看测试脚本与易错点（第 10、12 章）

## 2. 统一张量与空间约定（关键）
这部分是整个项目最容易出错的地方。后续章节默认遵循本章约定。

### 2.1 常用张量命名
- `V`：势场（potential），通常为 `(B, 1, H, W)`；数据集读出常见 `(B, H, W)`，训练脚本会补通道（如 `gf/train/train_latent_green.py:186`）
- `V_hat`：VAE 重建势场，形状 `(B, 1, H, W)`，见 `gf/models/vae.py:117`
- `g_obs`：观测 LDOS，形状 `(B, K, H, W)`，由数据集提供，见 `gf/data/dataset.py:153`
- `g_pred`：模型/代理预测的 LDOS（语义依上下文，重点看是否在线性域）
- `z`：VAE latent，形状 `(B, C, h, w)`，见 `gf/models/vae.py:92`
- `z_t`：加噪 latent，见 `gf/train/train_diffusion.py:323`
- `pred`：diffusion 输出（具体是 `eps` / `v` / `x0` 取决于 `prediction_type`），见 `gf/train/train_diffusion.py:325`
- `x0_pred`：从 diffusion 输出还原的 clean latent 估计，见 `gf/train/train_diffusion.py:347`

### 2.2 空间（域）约定：线性域 vs 观测域
关键函数在 `gf/utils/ldos_transform.py`：
- `ldos_obs_from_linear(...)`：线性 LDOS -> 数据集观测域，`gf/utils/ldos_transform.py:68`
- `ldos_linear_from_obs(...)`：观测域 -> 线性 LDOS，`gf/utils/ldos_transform.py:83`
- `force_linear_ldos_mode(...)`：强制脚本以“线性比较模式”运行（关闭/调整部分变换配置），`gf/utils/ldos_transform.py:98`

当前代码约定：
- `V` 始终是线性物理量（不走 `ldos_transform`）
- `g_obs` 默认是数据集观测域（可能经过 log/scale 等变换），由 `GFDataset` 在 `__getitem__` 中返回，见 `gf/data/dataset.py:153`
- `LatentGreen.forward()` 返回的 `g_pred` 是线性域 surrogate LDOS，文档字符串已明确，见 `gf/models/latent_green.py:142`
- `LatentGreen.loss()` 内部会把线性 `g_pred` 映射到观测域后再与 `g_obs` 比较，见 `gf/models/latent_green.py:191`

### 2.3 latent scale 约定（Diffusion 训练/推理）
配置在 `diffusion.training.latent_scale`，见 `gf/configs/default.yaml:277`。

支持模式（训练端逻辑在 `gf/train/train_diffusion.py:227`, `gf/train/train_diffusion.py:306`）：
- `none`：不缩放 latent
- `fixed`：按固定标量缩放
- `auto`：按样本自身 `std` 做逐样本缩放（训练时逐样本）

当前实现的关键一致性点（本版已修复并在文档中强调）：
- 训练时 `auto` 使用逐样本 `z.std(dim=(1,2,3), keepdim=True)`，见 `gf/train/train_diffusion.py:306`
- 验证/推理端 `TeacherSampler.unscale_factor` 现支持标量或 tensor（逐样本反缩放），定义于 `gf/inference/teacher_sampler.py:50`，使用于 `gf/inference/teacher_sampler.py:90`
- `train_diffusion.py` 验证采样与 `test_diffusion.py` 都会为 `auto` 模式设置逐样本反缩放因子，见 `gf/train/train_diffusion.py:541`、`gf/test_diffusion.py:188`

## 3. 数据生成与数据集（`gf/data`）
### 3.1 `GFDataset`：数据入口（不是 `PotentialDataset`）
当前实际数据集类是 `GFDataset`，定义于 `gf/data/dataset.py:19`。

职责：
- 管理缓存 / 分片缓存加载
- 调用势场采样器生成 `V`
- 调用 `KPMForward` 计算理想 LDOS
- 调用退化管线生成 `g_obs`
- 按配置应用 `ldos_transform`

返回格式在 `__getitem__` 文档中已写明，见 `gf/data/dataset.py:153`：
- `V`: `(H, W)`
- `g_obs`: `(K, H, W)`

### 3.2 `KPMForward.compute_ldos` 的两条路径
`KPMForward` 定义于 `gf/data/kpm_forward.py:6`。

核心接口：
- `compute_ldos(...)`：`gf/data/kpm_forward.py:44`
- `_compute_ldos_direct(...)`：`gf/data/kpm_forward.py:105`
- `_build_system(...)`：`gf/data/kpm_forward.py:133`

两条计算路径：
- KPM 路径：优先用于较大系统（需要 `kwant`），在 `compute_ldos` 中构建 `kwant.kpm.SpectralDensity`
- Direct inverse 路径：当 `direct_inverse.enabled=True` 且规模较小时走 `_compute_ldos_direct(...)`

### 3.3 `square` vs `graphene` 输出映射约定
图构建发生在 `_build_system(...)`，见 `gf/data/kpm_forward.py:133`。

关键实现点：
- `square` / `square_lattice`：按常规方格点构图
- `graphene` / `honeycomb`：使用 `kwant.lattice.honeycomb` 构造 A/B 子晶格

本版修复后的行为：
- 在 KPM 路径和 direct 路径中，graphene 模式对于同一 `(x, y)` 标签的多个 site（如 A/B 子晶格）采用聚合写入（`+=`），而不是覆盖写入。
- KPM 路径聚合：`gf/data/kpm_forward.py:100`, `gf/data/kpm_forward.py:102`
- Direct 路径聚合：`gf/data/kpm_forward.py:126`, `gf/data/kpm_forward.py:129`

### 3.4 当前实现限制（graphene 分支）
`graphene` 分支在 `_build_system` 中对部分方向依赖物理项采用简化构造，当前会显式给出 warning，见 `gf/data/kpm_forward.py:151`。

需要注意：
- `SOC / 磁场 / NNN` 在 graphene 分支可能仅近似表达（并非完整方向依赖实现）
- 若你做严格物理对比，需要单独验证该分支是否满足你的目标模型

### 3.5 关键配置入口（数据/物理）
`gf/configs/default.yaml` 中常用块：
- `project`: `gf/configs/default.yaml:1`
- `paths`: `gf/configs/default.yaml:10`
- `data`: `gf/configs/default.yaml:15`
- `potential_sampler`: `gf/configs/default.yaml:46`
- `physics`: `gf/configs/default.yaml:76`
- `degradation`: `gf/configs/default.yaml:100`

## 4. VAE（潜空间编码器/解码器）
### 代码定位
- 类定义：`gf/models/vae.py:24`
- `encode(...)`：`gf/models/vae.py:92`
- `decode(...)`：`gf/models/vae.py:108`
- `forward(...)`：`gf/models/vae.py:117`
- `loss(...)`：`gf/models/vae.py:127`
- `reparameterize(...)`：`gf/models/vae.py:152`

### 4.1 作用
- 将势场 `V` 编码到潜空间 `z`
- 将 `z` 解码回 `V_hat`
- 为 diffusion 提供低维生成空间

### 4.2 结构概述（当前实现）
编码器/解码器是 CNN + ResBlock 结构，ResBlock 在同文件顶部定义。

主要结构特征：
- 编码端按 `latent_downsample` 做下采样，输出 latent 特征
- `mode="vae"` 时编码器输出通道会被拆成 `mu/logvar`
- 解码端使用 `ConvTranspose2d` 上采样恢复到原分辨率

### 4.3 `mode=ae` / `mode=vae` 差异
在 `encode(...)` 中体现，见 `gf/models/vae.py:92`：
- `mode="vae"`：返回 `(z, mu, logvar)`，其中 `z` 通过 `reparameterize(...)` 采样得到
- `mode="ae"`：返回 `(h, None, None)`（无 KL 项）

在 `loss(...)` 中体现，见 `gf/models/vae.py:127`：
- 只有 `mode="vae"` 且 `mu/logvar` 存在时才计算 KL
- `mode="ae"` 时 `kl_loss` 为 0

### 4.4 输入输出接口约定
- `encode(V)`：输入支持 `(B,H,W)` 或 `(B,1,H,W)`（3D 会自动补通道），见 `gf/models/vae.py:92`
- `decode(z)`：输入 `(B,C,h,w)`，输出 `(B,1,H,W)`，见 `gf/models/vae.py:108`
- `forward(V)`：等价于 `decode(encode(V).z)`，返回 `(V_hat, mu, logvar)`，见 `gf/models/vae.py:117`

### 4.5 重建损失与 KL
`VAE.loss(...)` 当前支持的重建损失类型（配置 `vae.recon_loss_type`）：
- `mse`
- `l1`
- `log_cosh`

实现见 `gf/models/vae.py:127`。

KL 项形式（VAE 模式）为：
- `0.5 * mean(exp(logvar) + mu^2 - 1 - logvar)`，见 `gf/models/vae.py:144`

### 4.6 配置说明（重点：decoder 配置已生效）
VAE 配置块位于 `gf/configs/default.yaml:133`。

当前实现会读取：
- `vae.mode`
- `vae.latent_downsample`
- `vae.latent_channels`
- `vae.encoder.*`
- `vae.decoder.*`
- `vae.kl.weight`
- `vae.recon_loss_type`

本版同步的关键行为变化：
- `decoder.*` 配置项已独立生效，不再隐式沿用 `encoder.*`
- 读取位置：`gf/models/vae.py:39`, `gf/models/vae.py:40`, `gf/models/vae.py:41`
- 解码器构建使用这些字段：`gf/models/vae.py:74`, `gf/models/vae.py:77`, `gf/models/vae.py:78`, `gf/models/vae.py:86`, `gf/models/vae.py:87`

### 4.7 测试与评估建议（deterministic 重构）
`test_vae.py` 当前已采用 deterministic 评估路径：
- 先 `vae.encode(...)`
- 若存在 `mu` 则使用 `mu` 作为评估 latent
- 再 `vae.decode(z_eval)`

实现见 `gf/test_vae.py:97`, `gf/test_vae.py:99`。

这可以避免 `mode="vae"` 下重参数化随机性导致的测试重构抖动。

## 5. 条件编码器（`ConditionEncoder`）
### 代码定位
- 类定义：`gf/models/condition_encoder.py:5`
- `forward(...)`：`gf/models/condition_encoder.py:57`
- `mode` 配置读取：`gf/models/condition_encoder.py:16`
- `map` 分支输出：`gf/models/condition_encoder.py:87`

### 5.1 作用
将条件观测（通常是 `g_obs`）编码成 diffusion 可用的条件表示：
- token-like 序列（用于 cross-attn）
- feature map（用于 concat）

### 5.2 模式（按当前实现）
当前代码支持三类路径（由 `mode` 控制）：
- `energy_seq`：按能量维拆分成序列 token，见 `gf/models/condition_encoder.py:22`, `gf/models/condition_encoder.py:78`
- `map`：输出 `(B, C_latent_like, h, w)` 条件图，见 `gf/models/condition_encoder.py:49`, `gf/models/condition_encoder.py:87`
- 其他 token-like 路径（默认 `token`）：将 backbone 特征展平后投影到 `(B, L, E)`

### 5.3 与 `LatentDiffusion.cond_mode` 的契约（必须匹配）
`LatentDiffusion` 在初始化时会检查条件编码器输出形态与 `cond_mode` 是否匹配，见 `gf/models/diffusion.py:131`, `gf/models/diffusion.py:146`, `gf/models/diffusion.py:150`。

契约如下：
- `cond_mode="concat"` 必须搭配 `condition_encoder.mode="map"`
- `cond_mode != "concat"`（cross-attn 模式）不应搭配 `mode="map"`

当前实现会在初始化阶段直接抛出 `ValueError`，避免运行到 `MultiheadAttention` 时才崩溃。

## 6. LatentGreen（潜空间物理代理）
### 代码定位
- 类定义：`gf/models/latent_green.py:64`
- `forward(...)`：`gf/models/latent_green.py:142`
- `loss(...)`：`gf/models/latent_green.py:191`
- `_alpha_sigma(...)`：`gf/models/latent_green.py:312`
- `_ldos_from_psi(...)`：`gf/models/latent_green.py:332`
- `_apply_hamiltonian(...)`：`gf/models/latent_green.py:335`
- `residual_loss(...)`：`gf/models/latent_green.py:349`

### 6.1 角色定位
`LatentGreen` 是潜空间物理代理，用于近似 `z -> LDOS`，常用于：
- 独立训练 surrogate（`train_latent_green.py`）
- 作为 diffusion 的物理损失/引导辅助模块（训练或推理）

### 6.2 `forward` 输出语义（重点）
`LatentGreen.forward(...)` 返回的是线性域 surrogate LDOS，不是观测域数据，见 `gf/models/latent_green.py:142`。

当 `return_psi=True` 时返回：
- `g_pred`
- `psi_real`
- `psi_imag`
- `src`

其中 `g_pred` 由 `_ldos_from_psi` 构造：
- `g_pred = psi_real^2 + psi_imag^2`，见 `gf/models/latent_green.py:332`

这意味着它是一个非负 surrogate 表示，不应直接等同于标准复 Green 函数本体。

### 6.3 `loss`：线性预测 vs 观测目标的比较约定
`LatentGreen.loss(...)` 的接口契约已在 docstring 明确，见 `gf/models/latent_green.py:191`：
- 输入 `g_pred`：线性 LDOS
- 输入 `g_obs`：数据集观测域（可能含 `ldos_transform`）
- 函数内部会调用 `ldos_obs_from_linear(...)` 做空间转换后再比较

当前返回项包含：
- `loss`
- `data_loss`
- `fft_loss`
- `psd_loss`
- `stats_loss`
- `linear_scale_loss`
- `ms_loss`
- `residual_loss`

`psd_loss` 已显式返回，见 `gf/models/latent_green.py:305`。

### 6.4 `residual_loss` 的支持边界（已加保护）
当前 `residual_loss` 的离散哈密顿量实现是 square-lattice 风格（4 邻域 stencil），见 `_apply_hamiltonian(...)`：`gf/models/latent_green.py:335`。

因此它只对以下配置物理一致：
- square lattice
- 固定标量 hopping（不是随机范围）

本版修复后的行为：
- 若配置不满足支持边界且 residual 权重启用，会在初始化时 warning，并将 residual 项安全置零，见 `gf/models/latent_green.py:103`, `gf/models/latent_green.py:108`
- `residual_loss(...)` 在不支持配置下返回零，避免错误物理约束，见 `gf/models/latent_green.py:350`

### 6.5 noisy latent training 与 `_alpha_sigma`
`LatentGreen` 支持在潜空间做噪声增强训练（`add_noise(...)` / `_alpha_sigma(...)`），见 `gf/models/latent_green.py:176`, `gf/models/latent_green.py:312`。

本版同步的关键修复：
- `cosine` 调度下使用 `sqrt(alpha_bar_t)` 作为信号系数（与 `z_t = alpha * z + sigma * eps` 参数化一致）
- 代码位置：`gf/models/latent_green.py:318`

### 6.6 `train_latent_green.py` 中损失聚合（含 `psd_loss`）
训练脚本定义于 `gf/train/train_latent_green.py:25`。

当前主训练循环会：
1. 从 VAE 得到 `z`
2. 视配置决定是否对 `z` 加噪
3. 调用 `LatentGreen.forward(..., return_psi=True)`
4. 调用 `residual_loss(...)` 与 `loss(...)`
5. 在训练脚本中按权重重新聚合总损失（含 warmup）

关键修复点：`psd_loss` 已被纳入实际优化目标，见 `gf/train/train_latent_green.py:216`, `gf/train/train_latent_green.py:222`；日志也会打印 `psd`，见 `gf/train/train_latent_green.py:364`。

## 7. LatentDiffusion（条件潜空间扩散）
### 代码定位
- 类定义：`gf/models/diffusion.py:106`
- `forward(...)`：`gf/models/diffusion.py:223`
- `unpatchify(...)`：`gf/models/diffusion.py:207`
- `predict_eps(...)`：`gf/models/diffusion.py:273`
- `predict_x0(...)`：`gf/models/diffusion.py:287`
- `step(...)`：`gf/models/diffusion.py:298`
- `get_alpha_sigma(...)`：`gf/models/diffusion.py:330`

### 7.1 模型定位
`LatentDiffusion` 是在 VAE 潜空间上运行的条件 diffusion（DiT 风格 Transformer）。

输入：
- `z_t`: noisy latent
- `t`: diffusion timestep
- `cond_input`: 条件输入（通常是 `g_obs`，也可为已编码条件）

输出：
- 语义由 `prediction_type` 决定（`eps` / `v` / `x0`）

### 7.2 条件分支（concat / cross-attn）
条件路径由 `cond_mode` 控制，配置读取位置：`gf/models/diffusion.py:131`。

当前实现：
- `concat` 模式：将 `z_t` 与条件 map 在通道维拼接后送入 `x_embedder`，见 `gf/models/diffusion.py:230`
- cross-attn 模式：将条件编码为 token 序列，通过 DiT block 中的 cross attention 使用，见 `gf/models/diffusion.py:157`, `gf/models/diffusion.py:172`
- `use_green_attn` 开启时，会额外调用 `latent_green(z_t, t)` 生成代理条件并拼接 token，见 `gf/models/diffusion.py:132`, `gf/models/diffusion.py:241`

### 7.3 patch embedding 与 `unpatchify`
当前实现显式暴露 `unpatchify(...)`，见 `gf/models/diffusion.py:207`。

说明：
- patch embedding 通过 `self.x_embedder = Conv2d(..., kernel_size=patch_size, stride=patch_size)` 实现，见 `gf/models/diffusion.py:158`
- Transformer 输出在 `final_layer` 后通过 `unpatchify(...)` 恢复成 `(B, C, H, W)` latent 形状

### 7.4 `prediction_type = eps / v / x0` 契约
训练端 target 构造在 `gf/train/train_diffusion.py:327`，模型端统一转换接口在：
- `predict_eps(...)`: `gf/models/diffusion.py:273`
- `predict_x0(...)`: `gf/models/diffusion.py:287`

当前代码实现中，这三种参数化在代数上是自洽的；文档层面需要记住：
- `pred` 的语义由配置 `diffusion.training.prediction_type` 决定（默认配置位置 `gf/configs/default.yaml:247`）
- 训练、验证和推理都必须使用同一 `prediction_type`

### 7.5 `step()` 与稀疏时间步 `t_prev`（本版关键修复）
`LatentDiffusion.step(...)` 现支持显式 `t_prev` 参数，见 `gf/models/diffusion.py:304`。

这用于支持 `TeacherSampler` 的稀疏时间步采样（例如 200 步而不是逐步 1000 步）：
- 若不传 `t_prev`，默认用 `(t - 1).clamp_min(0)`，见 `gf/models/diffusion.py:308`
- 传入 `t_prev` 时按实际稀疏时间表更新，避免“采样步长和公式步长不一致”

### 7.6 `get_alpha_sigma` 端点与时间步范围
`get_alpha_sigma(...)` 见 `gf/models/diffusion.py:330`。

当前行为：
- 会对输入 `t` 做 clamp 到 `[0, T-1]`，见 `gf/models/diffusion.py:331`
- 支持 `cosine` 和线性风格调度（依据 `self.schedule`）

这使得推理端即使意外传入越界时间步，也会在模型内部收敛到合法范围。

### 7.7 `eta > 0` 的采样更新
`step()` 中的随机项使用 DDIM 风格的 `sigma_eta` / `dir_coeff` 更新（而非简单直接加噪），见 `gf/models/diffusion.py:320`, `gf/models/diffusion.py:324`。

这也是本版相对旧行为的重要修复之一。

## 8. TeacherSampler 与推理增强（`gf/inference/teacher_sampler.py`, `gf/guidance/*`）
### 代码定位
- `TeacherSampler`：`gf/inference/teacher_sampler.py:13`
- `TeacherSampler.__init__`：`gf/inference/teacher_sampler.py:17`
- `TeacherSampler.sample`：`gf/inference/teacher_sampler.py:52`
- `TeacherSampler._sample_steps`：`gf/inference/teacher_sampler.py:105`
- `LatentGuidance`：`gf/guidance/latent_guidance.py:5`
- `RestartSampler`：`gf/guidance/restart.py:7`

### 8.1 角色定位
`TeacherSampler` 是“高质量教师采样器”封装，负责在推理端把以下组件串起来：
- `LatentDiffusion`
- `VAE`
- `ConditionEncoder`
- `LatentGreen`
- `LatentGuidance`
- （可选）`RestartSampler` + KPM 检查

### 8.2 `config` 处理与 `force_linear_ldos_mode`（副作用已消除）
`TeacherSampler.__init__` 内部会对传入 `config` 做深拷贝，然后调用 `force_linear_ldos_mode(...)`，见 `gf/inference/teacher_sampler.py:26`, `gf/inference/teacher_sampler.py:27`。

含义：
- `TeacherSampler` 会在内部配置上使用线性 LDOS 模式（便于与 `LatentGreen`/物理路径一致）
- 不会再原地污染调用方传入的 `config`（这是本版修复点）

### 8.3 稀疏时间步采样（本版关键修复）
`TeacherSampler.sample(...)` 当前使用训练一致的时间步域 `[0, T-1]`，时间表生成见 `gf/inference/teacher_sampler.py:74`。

关键修复点：
- 不再把 `t=T` 当作采样起点
- `_sample_steps(...)` 会为每步构造 `t_prev_batch` 并显式传入 `diffusion.step(..., t_prev=...)`，见 `gf/inference/teacher_sampler.py:113`, `gf/inference/teacher_sampler.py:115`
- 最后一跳会明确落到 `t_prev=0`

### 8.4 `unscale_factor` 与 latent scale 一致性
`TeacherSampler.unscale_factor` 定义为 `Union[float, torch.Tensor]`，见 `gf/inference/teacher_sampler.py:50`。

在 `sample(...)` 中：
- 支持标量反缩放
- 支持逐样本 tensor 反缩放（用于 `latent_scale=auto`）
- 逻辑见 `gf/inference/teacher_sampler.py:90`

### 8.5 guidance / restart / KPM-check 的位置
在 `_sample_steps(...)` 中，采样流程大致为：
1. 调用 `diffusion.step(...)` 做一步 latent 更新
2. 若 guidance 启用且满足时间步条件，则应用 `LatentGuidance`
3. 若存在 restart mask，仅替换被 mask 样本的 latent

guidance 启用判定与时间条件在 `gf/inference/teacher_sampler.py:117` 附近。

### 8.6 `kwant` 缺失时的降级行为
`TeacherSampler.__init__` 会尝试构建 `KPMForward` + `RestartSampler`；若 `kwant` 不可用，则自动关闭 KPM check / restart 并给出 warning，见 `gf/inference/teacher_sampler.py:35`。

这保证主采样流程仍可运行，但会失去 KPM 校验与重启能力。

## 9. 训练脚本（分阶段）
### 9.1 `train_vae.py`
- 入口：`gf/train/train_vae.py:22`
- 作用：训练 `VAE`

当前实现中的关键点：
- resume 时会优先使用当前 run 的 checkpoint（若当前目录已有 `vae_step_*.pt`），再回退到 latest run，见 `gf/train/train_vae.py:99`, `gf/train/train_vae.py:107`, `gf/train/train_vae.py:108`
- 若因架构不兼容无法加载 checkpoint，会切回当前 run 目录并从头训练，见 `gf/train/train_vae.py:127`, `gf/train/train_vae.py:129`

### 9.2 `train_latent_green.py`
- 入口：`gf/train/train_latent_green.py:25`
- 作用：训练 `LatentGreen` surrogate，并可叠加残差/频域/统计约束

流程关键点：
- 从 `VAE.encode(V)` 得到 `z`，见 `gf/train/train_latent_green.py:188`
- 可选 noisy latent training，见 `gf/train/train_latent_green.py:190`
- `LatentGreen.forward(..., return_psi=True)` + `residual_loss(...)` + `loss(...)`，见 `gf/train/train_latent_green.py:203`
- 训练脚本中按 warmup 重组总损失，见 `gf/train/train_latent_green.py:220`

本版修复同步：
- `psd_loss` 已进入训练总损失，见 `gf/train/train_latent_green.py:222`

### 9.3 `train_diffusion.py`
- 入口：`gf/train/train_diffusion.py:31`
- 作用：训练条件潜空间 diffusion（可叠加 `x0_loss`、物理损失、一致性损失等）

关键流程（按代码顺序）：
1. 从 `VAE.encode(V)` 得到 `z`（优先使用 `mu`），见 `gf/train/train_diffusion.py:303`
2. 按 `latent_scale` 处理 `z`，见 `gf/train/train_diffusion.py:306`
3. 随机采样 `t` 并构造 `z_t = alpha_t * z + sigma_t * noise`，见 `gf/train/train_diffusion.py:313`, `gf/train/train_diffusion.py:323`
4. 调用 diffusion 得到 `pred`，见 `gf/train/train_diffusion.py:325`
5. 按 `prediction_type` 构造训练目标 `target`，见 `gf/train/train_diffusion.py:327`
6. 按需要从 `pred` 还原 `x0_pred`，见 `gf/train/train_diffusion.py:349`
7. 可选物理损失中通过 `LatentGreen` 计算 `g_pred_phys` 并映射到观测域比较，见 `gf/train/train_diffusion.py:380` 附近（连续代码段）

EMA 与验证采样：
- EMA 初始化与恢复逻辑在 `gf/train/train_diffusion.py:200` 段
- 验证采样器 `TeacherSampler` 构建于 `gf/train/train_diffusion.py:243`
- `latent_scale=auto` 下为 `val_sampler.unscale_factor` 设置逐样本 tensor，见 `gf/train/train_diffusion.py:541`, `gf/train/train_diffusion.py:546`

### 9.4 `train_student.py`
- 入口：`gf/train/train_student.py:24`
- 作用：学生模型蒸馏训练（直接 `g_obs -> V` 的轻量路径）

本文件不在本轮文档重点展开内部细节，但会在“最小工作流”和“模块关系”中保留位置，便于串联完整 pipeline。

## 10. 测试与评估脚本（`gf/test_*.py`）
### 10.1 `test_vae.py`
- 入口：`gf/test_vae.py:39`
- 当前 ckpt 查找逻辑优先当前 run，再回退 latest run，见 `gf/test_vae.py:17`, `gf/test_vae.py:18`
- 当前重构评估使用 deterministic `mu -> decode` 路径，见 `gf/test_vae.py:97`, `gf/test_vae.py:99`

### 10.2 `test_green.py`
- 入口：`gf/test_green.py:79`
- 会先 `force_linear_ldos_mode(config, ...)`，见 `gf/test_green.py:101`
- `g_obs` 会用 `ldos_linear_from_obs(...)` 还原到线性域后比较，见 `gf/test_green.py:183`

本版修复同步：
- 文档强调 `LatentGreen` 的 `g_pred` 已经是线性域，不应再对 `g_pred` 做观测域逆变换（脚本中已改正并加注释）

### 10.3 `test_latent_green.py`
- 入口：`gf/test_latent_green.py:60`
- 会强制线性评估模式，见 `gf/test_latent_green.py:69`
- 同时支持“模型空间（观测域）”与“物理空间（线性域）”指标，并使用 `align_pred / per_energy_affine` 做对齐，见 `gf/test_latent_green.py:209` 与 `gf/utils/loss_align.py:24`, `gf/utils/loss_align.py:34`

### 10.4 `test_diffusion.py`
- 入口：`gf/test_diffusion.py:61`
- 强制线性 LDOS 模式：`gf/test_diffusion.py:100`
- `latent_scale` 推理反缩放设置：`gf/test_diffusion.py:188`
- `phys_mse` 计算时会先把 `LatentGreen` 的线性输出映射到观测域再与 `g_obs` 比较，见 `gf/test_diffusion.py:238`, `gf/test_diffusion.py:246`, `gf/test_diffusion.py:260`

本版修复同步：
- `phys_mse` 不再混用线性域与观测域
- `Norm Cond` 分支的 VAE 编码路径与其它分支一致（优先使用 `mu`），见 `gf/test_diffusion.py:242`, `gf/test_diffusion.py:243`

### 10.5 测试脚本与空间一致性的经验法则
- 比较 `g_obs` 时，先确认目标是“观测域指标”还是“线性物理指标”
- `LatentGreen` 输出默认按线性域理解
- 用 `ldos_obs_from_linear(...)` / `ldos_linear_from_obs(...)` 做显式转换，不要隐式猜测

## 11. 默认配置与关键配置图谱（`gf/configs/default.yaml`）
### 11.1 顶层配置块（推荐先看）
- `project`: `gf/configs/default.yaml:1`
- `paths`: `gf/configs/default.yaml:10`
- `data`: `gf/configs/default.yaml:15`
- `potential_sampler`: `gf/configs/default.yaml:46`
- `physics`: `gf/configs/default.yaml:76`
- `degradation`: `gf/configs/default.yaml:100`
- `vae`: `gf/configs/default.yaml:133`
- `latent_green`: `gf/configs/default.yaml:161`
- `diffusion`: `gf/configs/default.yaml:213`
- `guidance`: `gf/configs/default.yaml:292`
- `validation`: `gf/configs/default.yaml:303`
- `student`: `gf/configs/default.yaml:319`
- `eval`: `gf/configs/default.yaml:345`

### 11.2 Diffusion 相关关键字段（常用）
在 `diffusion` 块中重点关注：
- `condition_encoder`：`gf/configs/default.yaml:216`
- `model.cond_mode`：`gf/configs/default.yaml:229`
- `training.prediction_type`：`gf/configs/default.yaml:247`
- `training.latent_scale`：`gf/configs/default.yaml:277`
- `sampler.steps` / `sampler.eta`：`gf/configs/default.yaml:289`, `gf/configs/default.yaml:290`

### 11.3 Green / 物理相关字段（常用）
在 `physics` 和 `latent_green` 块中重点关注：
- `physics.kpm.eta`：`gf/configs/default.yaml:83`
- `physics.hamiltonian.*`（晶格类型、hopping、化学势等）
- `latent_green.model.*`（loss 权重、残差项、对齐项等）

### 11.4 配置与实现契约提醒
- `diffusion.model.cond_mode` 与 `diffusion.condition_encoder.mode` 必须匹配（见第 5 章）
- `latent_scale=auto` 需要训练与推理都正确设置反缩放因子（见第 2.3、8.4、9.3、10.4）
- `LatentGreen residual_loss` 仅在 square + 固定 hopping 下物理一致（见第 6.4）

## 12. 模型兼容性与易错点（重点）
### 12.1 线性域 / 观测域混用
这是本项目最常见的错误来源。

典型规则：
- `LatentGreen.forward()` 输出线性域 `g_pred`（`gf/models/latent_green.py:142`）
- `g_obs` 通常来自数据集观测域（`gf/data/dataset.py:153`）
- 计算损失或指标前必须明确比较空间

推荐做法：
- 物理指标：先把 `g_obs` 转回线性域（`ldos_linear_from_obs`）
- 数据/训练指标：把 `g_pred` 映射到观测域（`ldos_obs_from_linear`）

### 12.2 `cond_mode` 与条件编码器输出形状不匹配
当前已在 `LatentDiffusion.__init__` 中增加显式校验（`gf/models/diffusion.py:146`, `gf/models/diffusion.py:150`），但仍建议在改配置时优先检查：
- `concat` <-> `map`
- `cross_attn` <-> token-like

### 12.3 diffusion 稀疏采样时间步与 `step()` 端点
本版已修复 `TeacherSampler` 与 `LatentDiffusion.step()` 的稀疏时间步契约：
- 采样时间步在训练域 `[0, T-1]`（`gf/inference/teacher_sampler.py:74`）
- `step(..., t_prev=...)` 支持显式前一时刻（`gf/models/diffusion.py:304`）

如果你替换采样器或自定义 schedule，必须保证：
- 时间步表和 `t_prev` 一致
- 最后一跳能落到 `t=0`

### 12.4 latent scale 误配导致推理偏差
尤其在 `latent_scale=auto` 下，若推理时只用单标量近似反缩放，可能与训练分布不一致。

当前推荐路径：
- 使用逐样本 `unscale_factor` tensor（`TeacherSampler` 已支持，见 `gf/inference/teacher_sampler.py:50`, `gf/inference/teacher_sampler.py:90`）

### 12.5 checkpoint 目录误用
多 run 并行时，容易把“当前 run”和“latest run”的 checkpoint 混用。

当前脚本行为（已优化）：
- `train_vae.py` 优先当前目录（`gf/train/train_vae.py:107`, `gf/train/train_vae.py:108`）
- `test_vae.py` 查找函数优先当前目录（`gf/test_vae.py:17`, `gf/test_vae.py:18`）

建议：
- 调试时显式传 `--ckpt_dir`
- 记录每次测试使用的 config 路径与 ckpt 路径

### 12.6 `kwant` 缺失带来的功能降级
影响范围：
- `KPMForward` 的 KPM 路径
- `TeacherSampler` 的 KPM-check / restart

当前行为：
- `TeacherSampler` 会自动禁用 KPM check 与 restart 并给出 warning（`gf/inference/teacher_sampler.py:35`）

### 12.7 graphene 相关物理简化
虽然 `KPMForward` 已修复 A/B 子晶格映射覆盖问题（改为聚合），但 graphene 分支仍存在简化 hopping 构造，见 `gf/data/kpm_forward.py:151`。

需要严格物理 fidelity 时：
- 建议单独验证 graphene 分支的构图和 hopping 是否满足目标方程

### 12.8 历史 checkpoint 兼容性提醒
本版代码对以下行为有修复/变化：
- VAE decoder 独立配置生效
- diffusion 稀疏采样端点与 `t_prev` 契约修复
- `TeacherSampler` 逐样本 `unscale_factor` 支持
- `test_diffusion.py` `phys_mse` 空间一致性修复
- `LatentGreen` residual 不支持场景下的安全降级

因此旧 checkpoint 在新脚本中的指标、采样风格可能与历史记录不完全可比。

## 13. 最小工作流（命令示例）
以下示例按“脚本入口”风格给出，具体路径可按你的环境调整。

### 13.1 一键分阶段运行（推荐入口）
`run_pipeline.py` 提供交互式/指定阶段运行入口，主函数见 `gf/run_pipeline.py:71`，并会调用 `force_linear_ldos_mode(...)`，见 `gf/run_pipeline.py:98`。

示例：
```bash
python -m gf.run_pipeline --config gf/configs/default.yaml --stages vae green diffusion
```

可选阶段（见 `gf/run_pipeline.py:74`）：
- `data`
- `vae`
- `green`
- `diffusion`
- `student`

### 13.2 单独训练 VAE
```bash
python -m gf.train.train_vae --config gf/configs/default.yaml
```

### 13.3 单独训练 LatentGreen
```bash
python -m gf.train.train_latent_green --config gf/configs/default.yaml
```

### 13.4 单独训练 Diffusion
```bash
python -m gf.train.train_diffusion --config gf/configs/default.yaml
```

### 13.5 测试脚本（常用）
```bash
python -m gf.test_vae --config gf/configs/default.yaml
python -m gf.test_green --config gf/configs/default.yaml
python -m gf.test_latent_green --config gf/configs/default.yaml
python -m gf.test_diffusion --config gf/configs/default.yaml
```

说明：
- 若你的项目环境使用“直接运行文件”风格（如 `python gf/test_diffusion.py`），也可沿用
- 建议测试时显式指定 `--ckpt_dir`，避免多 run 场景加载到错误 checkpoint

## 14. 训练/推理链路速查表（面向排错）
### 14.1 VAE 链路
`V -> VAE.encode -> z -> VAE.decode -> V_hat`

排错优先看：
- 输入形状是否为 `(B,1,H,W)`（`encode` 可容忍 3D）
- `mode=vae` 时测试是否使用 `mu` 做 deterministic 重构
- `decoder.*` 配置是否与你预期一致（本版已生效）

### 14.2 LatentGreen 链路
`z (+t) -> LatentGreen -> g_pred(linear)`

排错优先看：
- 是否错误把 `g_pred` 当成观测域数据
- `residual_loss` 是否在不支持配置下被正确降级为 0（square/fixed hopping 之外）
- `ldos_transform` 与损失比较空间是否一致

### 14.3 Diffusion + Teacher 采样链路
`g_obs -> TeacherSampler -> diffusion sample z -> (unscale) -> VAE.decode -> V`

排错优先看：
- `cond_mode` 与 `condition_encoder.mode` 契约
- `prediction_type` 与 checkpoint/训练配置是否一致
- `latent_scale` 反缩放因子是否正确
- `kwant` 缺失是否导致 restart/KPM-check 被禁用

## 15. 附：共享对齐/损失工具（`gf/utils/loss_align.py`）
该模块用于减少训练/测试脚本中重复的对齐与损失逻辑，主要函数：
- `loss_map(...)`：`gf/utils/loss_align.py:7`
- `per_energy_affine(...)`：`gf/utils/loss_align.py:24`
- `align_pred(...)`：`gf/utils/loss_align.py:34`

当前已在多个模块中复用：
- `gf/models/latent_green.py:13`
- `gf/train/train_latent_green.py:18`
- `gf/test_latent_green.py:31`
- `gf/train/train_diffusion.py:20`

这有助于减少“训练/评估逻辑漂移”的风险。

## 16. 总结
GreenDiff 当前实现可以理解为一个“分层生成 + 物理代理约束”的工程体系：
- 数据层：`GFDataset` + `KPMForward`
- 表示层：`VAE`
- 物理代理层：`LatentGreen`
- 生成层：`LatentDiffusion`
- 推理增强层：`TeacherSampler` + `Guidance/Restart`

如果你要修改或扩展项目，建议优先守住三条契约：
1. 空间/域契约（线性 LDOS vs 观测域）
2. 条件形状契约（`cond_mode` vs `condition_encoder.mode`）
3. 采样时间步契约（`timesteps` 与 `step(..., t_prev=...)`）

守住这三条，绝大多数“训练能跑但结果奇怪”的问题都能提前规避。
