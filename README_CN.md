# GreenDiff 模型总文档（基于当前代码实现）

## 源码基线
本文件基于以下代码实现整理：

- `gf/models/vae.py:24`
- `gf/models/diffusion.py:106`
- `gf/models/latent_green.py:56`
- `gf/models/condition_encoder.py:5`
- `gf/models/student.py:23`
- `gf/inference/teacher_sampler.py:11`
- `gf/guidance/latent_guidance.py:4`
- `gf/guidance/restart.py:6`
- `gf/train/train_vae.py:21`
- `gf/train/train_latent_green.py:23`
- `gf/train/train_diffusion.py:29`
- `gf/train/train_student.py:23`
- `gf/configs/default.yaml`

## 说明
- `gf/configs/default.yaml` 的中文注释在当前终端显示存在编码乱码，但配置数值本身可正常读取和使用。
- `gf/models/__init__.py` 当前为空，项目使用显式模块路径导入模型类。

## 1. 总体架构（模型关系图）
GreenDiff 当前的核心模型链路可以概括为：

`g_obs (LDOS观测)` -> `ConditionEncoder` -> `LatentDiffusion (DiT)` -> `z` -> `VAE.decode` -> `V`

同时存在一条物理代理链路：

`z` -> `LatentGreen` -> `g_pred`

训练时的主要依赖关系：

- `VAE` 先训练，提供潜变量空间。
- `LatentGreen` 使用 `VAE.encode(V)` 的潜变量训练，学习 `z -> g` 代理。
- `LatentDiffusion` 在潜空间做条件扩散，训练时可调用 `LatentGreen` 给 `x0` 增加物理损失。
- `StudentModel` 蒸馏 `TeacherSampler` 的输出，并可额外使用 `VAE + LatentGreen` 的物理一致性损失。

## 2. 模型清单（按代码实际定义）
### 2.1 核心模型类
- `VAE`：潜空间压缩与重建，定义于 `gf/models/vae.py:24`
- `LatentDiffusion`：条件潜空间扩散模型（DiT风格 Transformer），定义于 `gf/models/diffusion.py:106`
- `LatentGreen`：潜空间物理代理（近似 `z -> LDOS`），定义于 `gf/models/latent_green.py:56`
- `ConditionEncoder`：条件编码器（LDOS -> token/map），定义于 `gf/models/condition_encoder.py:5`
- `StudentModel`：轻量学生网络（直接 `g_obs -> V`），定义于 `gf/models/student.py:23`

### 2.2 辅助/推理模块（不是“模型定义”，但会直接调用模型）
- `TeacherSampler`：教师采样器，定义于 `gf/inference/teacher_sampler.py:11`
- `LatentGuidance`：潜空间物理引导，定义于 `gf/guidance/latent_guidance.py:4`
- `RestartSampler`：KPM校验与重启采样，定义于 `gf/guidance/restart.py:6`

## 3. 统一张量约定（文档后续默认使用）
- `V`：势场（potential），形状通常为 `(B, 1, H, W)`；数据集读出时常见 `(B, H, W)`，训练脚本会 `unsqueeze(1)`
- `g_obs`：观测 LDOS，形状 `(B, K, H, W)`
- `z`：VAE 潜变量，形状 `(B, C_latent, h, w)`
- `t`：时间步，形状 `(B,)`，整数 `long`
- 默认配置中 `resolution=64`，`K=16`，`latent_downsample=2`，因此常见 `z` 空间大小为 `32x32`

## 4. VAE（潜空间编码器/解码器）
### 代码定位
- 类定义：`gf/models/vae.py:24`
- `encode`：`gf/models/vae.py:87`
- `decode`：`gf/models/vae.py:103`
- `loss`：`gf/models/vae.py:122`

### 4.1 作用
- 将势场 `V` 压缩到潜空间 `z`
- 将潜变量 `z` 重建回势场 `V_hat`
- 为扩散模型提供更低维、更平滑的生成空间

### 4.2 结构概述
VAE 使用 CNN + ResBlock 的编码器/解码器结构，内部 `ResBlock` 定义在同文件 `gf/models/vae.py:8`。

编码器流程（实际代码逻辑）：
- `Conv2d(1 -> base_channels)` + `SiLU`
- 多个 `ResBlock`
- 根据 `latent_downsample` 进行 1 次或 2 次 stride=2 下采样
- 输出通道数为：
  - `latent_channels`（AE 模式）
  - `2 * latent_channels`（VAE 模式，用于 `mu/logvar`）

解码器流程（实际代码逻辑）：
- `Conv2d(latent_channels -> base_channels * mult_last)`
- 多个 `ResBlock`
- 通过 `ConvTranspose2d` 逐步上采样
- 最终 `Conv2d(... -> 1)` 输出重建势场

### 4.3 关键配置（代码实际读取）
读取自 `config["vae"]` 与 `config["data"]`，包括：
- `vae.mode`：`ae` 或 `vae`
- `vae.latent_downsample`
- `vae.latent_channels`
- `vae.encoder.base_channels`
- `vae.encoder.num_res_blocks`
- `vae.encoder.dropout`
- `vae.kl.weight`
- `vae.recon_loss_type`
- `vae.recon_log_cosh_eps`
- `data.resolution`

重要实现事实：
- 当前 `VAE` 实现主要读取 `vae.encoder.*`，没有单独读取 `vae.decoder.*` 的参数。
- `default.yaml` 里虽然有 `vae.decoder` 段，但在当前 `VAE` 构造函数中未实际使用。

### 4.4 输入输出与接口
`encode(V)`：
- 输入 `V` 支持 `(B,H,W)` 或 `(B,1,H,W)`，3维时会自动补通道
- 返回 `(z, mu, logvar)`
- 在 `mode="vae"` 时，`z = mu + eps * std`
- 在 `mode="ae"` 时，返回 `(h, None, None)`

`decode(z)`：
- 输入 `(B,C,h,w)`
- 输出 `(B,1,H,W)`

`forward(V)`：
- 等价于 `decode(encode(V).z)`

### 4.5 损失函数
`loss(V, V_hat, mu, logvar)` 返回字典：
- `loss`
- `recon_loss`
- `kl_loss`

支持重建损失类型：
- `l1`
- `mse`
- `log_cosh`

KL 项仅在 `mode="vae"` 且 `mu/logvar` 存在时生效：
- `kl = 0.5 * mean(exp(logvar) + mu^2 - 1 - logvar)`

### 4.6 约束与限制（代码硬性检查）
- `latent_channels` 必须是 `4` 或 `8`
- `resolution` 必须是 `64` 或 `256`
- `latent_downsample` 必须是 `2` 或 `4`

## 5. ConditionEncoder（条件编码器）
### 代码定位
- 类定义：`gf/models/condition_encoder.py:5`
- `forward`：`gf/models/condition_encoder.py:57`

### 5.1 作用
将 `g_obs`（LDOS观测）编码为扩散模型可用的条件表示。支持三种输出形态：
- token 序列
- feature map（用于 concat 条件）
- energy sequence（按能量切片逐个编码）

### 5.2 模式说明（非常关键）
`mode="map"`：
- 输出形状 `(B, latent_channels, h, w)`
- 用于 `LatentDiffusion` 的 `cond_mode="concat"` 路径最合适（默认配置就是这种）

`mode="token"`（默认语义）：
- 输出形状 `(B, L, token_dim)`
- 用于 `LatentDiffusion` 的 cross-attention 条件

`mode="energy_seq"`：
- 将输入 `(B,K,H,W)` reshape 成 `(B*K,1,H,W)` 独立编码
- 输出 `(B,K,token_dim)`
- 更强调“按能量通道的序列条件”

### 5.3 坐标与归一化
支持：
- `normalize`：按通道做空间均值/标准差归一化
- `use_coords`：在输入拼接 `x,y` 坐标通道（仅非 `energy_seq` 模式）

### 5.4 下采样逻辑（代码实际行为）
编码 backbone 会按层构建卷积块，并“尝试”下采样到与 VAE 潜空间类似的尺度。
- 每层计算一个 `current_ds = 2^(i+1)`
- 若尚未超过目标 `vae.latent_downsample`，则使用 stride=2
- 否则 stride=1

这使得 `ConditionEncoder` 的空间输出在 `mode="map"` 下更容易与潜空间分辨率兼容。

### 5.5 输出维度兼容性（务必注意）
当前 `LatentDiffusion` 的 cross-attn 使用 `nn.MultiheadAttention(hidden_size, ...)`，没有指定 `kdim/vdim`，因此：
- 当使用 cross-attn（`cond_mode != "concat"`）且条件是 token 时，`ConditionEncoder` 的 token 维度必须等于 `diffusion.model.hidden_size`

这点在 `gf/test_dit.py` 中也有明确注释示例（`token_dim == hidden_size`）。

### 5.6 默认配置下的实际角色
`default.yaml` 中默认：
- `diffusion.model.cond_mode: concat`
- `diffusion.condition_encoder.mode: map`

因此默认是“输出条件特征图，然后和 `z_t` 通道拼接”的路线。

## 6. LatentGreen（潜空间物理代理）
### 代码定位
- 类定义：`gf/models/latent_green.py:56`
- `forward`：`gf/models/latent_green.py:115`
- `add_noise`：`gf/models/latent_green.py:146`
- `loss`：`gf/models/latent_green.py:161`
- `residual_loss`：`gf/models/latent_green.py:341`

### 6.1 作用
`LatentGreen` 学习从潜变量 `z` 预测 LDOS（或其代理量）：
- 输入：`z`
- 输出：`g_pred`（形状 `(B,K,H,W)`）

它在系统中的用途有三类：
- 训练时作为 `LatentDiffusion` 的物理损失代理
- 推理时作为 `LatentGuidance` 的可微引导器
- 评估时作为“快速物理一致性”近似检查器

### 6.2 结构概述
内部主要组件：
- `TimestepEmbedder`（可选时间步嵌入）
- 带时间注入的 `ResBlock`
- 潜空间到图像空间的 0~2 次上采样
- 两个输出头：
  - `psi_out`：输出 `2K` 通道（实部 + 虚部）
  - `src_out`：输出 `K` 通道源项

最终 `g_pred = psi_real^2 + psi_imag^2`（`_ldos_from_psi`）

### 6.3 时间步与上采样
`LatentGreen` 支持 `use_timestep=True`，这使它可在“噪声潜变量”场景下训练和使用。

上采样次数由 `vae.latent_downsample` 决定：
- `1` -> 0次上采样
- `2` -> 1次上采样
- `4` -> 2次上采样

因此默认配置（`latent_downsample=2`）下，`z` 从 `32x32` 上采样到 `64x64` 与 `g_obs` 对齐。

### 6.4 `forward` 的返回
`forward(z, t=None, return_psi=False)`：
- 默认返回 `g_pred`
- 当 `return_psi=True` 时返回 `(g_pred, psi_real, psi_imag, src)`

这使训练脚本能额外计算物理残差损失（`residual_loss`）。

### 6.5 `add_noise`（LatentGreen 专用噪声训练）
`add_noise(z, t)` 会根据 `latent_green.noisy_latent_training` 的 `T` 和 `schedule` 生成：
- `z_t`
- `alpha`
- `sigma`

这让 `LatentGreen` 不只看“干净潜变量”，还能适应扩散过程中的中间噪声潜变量。

### 6.6 `loss`（多项损失组合）
`loss(g_pred, g_obs, residual_loss=None)` 返回多个分量，核心特点如下：

数据项（`data_loss`）：
- 支持 `mse / l1 / huber / log_cosh`
- 可选对 `g_pred` 做 log 变换（当 `data.ldos_transform.log.enabled` 时）
- 可选 `per_energy_affine`（对每个能量通道做线性校准）
- 可选空间对齐（`energy_align`，通过 `torch.roll` 搜索最优平移）

频域项：
- `fft_loss`：对幅度谱做 log-L1（`_fft_loss`）
- `psd_loss`：对功率谱密度做 log-L1（内部 `_psd_loss`）

统计项：
- `stats_loss`：匹配每个能量通道的均值和标准差
- `linear_scale_loss`：约束线性尺度一致性（对数或线性域，取决于 LDOS 变换配置）

多尺度项：
- `ms_loss`：对 `avg_pool2d` 下采样后的结果做损失

物理残差项：
- `residual_loss`：通过离散哈密顿量残差约束 `psi_real/psi_imag/src` 和 `V` 的一致性

### 6.7 `residual_loss` 的物理含义（代码实现层面）
`residual_loss` 使用 `_apply_hamiltonian` 构造邻域差分算子（上下左右四邻域），结合：
- `physics.hamiltonian.t`
- `physics.hamiltonian.mu`
- `physics.kpm.eta`
- `data.energies`

形成实部/虚部残差并取平方均值。

这不是“精确 KPM”，而是用于训练中的可微近似物理约束。

### 6.8 训练脚本中的额外损失重组（很重要）
在 `train_latent_green.py`（`gf/train/train_latent_green.py:23`）里，`model.loss(...)` 返回后会被再次重组总损失：
- `fft_loss`、`stats_loss`、`ms_loss` 会乘上 `aux_scale`（辅助损失 warmup）
- `linear_scale_loss` 与 `residual_loss` 不受 `aux_scale` 缩放
- 这意味着最终训练目标以脚本重组为准，而不完全等同于 `LatentGreen.loss` 内部直接返回的 `loss`

## 7. LatentDiffusion（DiT 风格潜空间条件扩散）
### 代码定位
- 类定义：`gf/models/diffusion.py:106`
- `forward`：`gf/models/diffusion.py:213`
- `predict_eps`：`gf/models/diffusion.py:263`
- `predict_x0`：`gf/models/diffusion.py:277`
- `step`：`gf/models/diffusion.py:288`
- `get_alpha_sigma`：`gf/models/diffusion.py:305`

### 7.1 作用
给定条件 `g_obs`，在 VAE 潜空间中进行扩散采样，最终生成潜变量 `z`，再由 `VAE.decode(z)` 重建势场 `V`。

### 7.2 内部子模块（同文件定义）
- `modulate`：AdaLN 风格调制函数，`gf/models/diffusion.py:9`
- `TimestepEmbedder`：时间嵌入，`gf/models/diffusion.py:12`
- `DiTBlock`：Transformer block（自注意力 + 可选交叉注意力 + MLP + AdaLN），`gf/models/diffusion.py:39`
- `FinalLayer`：最终输出层（AdaLN + Linear），`gf/models/diffusion.py:90`

### 7.3 条件输入路径（两种模式）
`cond_mode="concat"`（默认配置）：
- `ConditionEncoder` 输出 map
- 与 `z_t` 按通道拼接
- 再做 patch embedding
- 不使用 cross-attention

`cond_mode != "concat"`（cross-attn 路径）：
- `ConditionEncoder` 输出 token 序列 `(B,L,D)`
- `DiTBlock` 中启用 cross-attention
- `z_t` 仅自身做 patch embedding
- 可选 `use_green_attention=True`，把 `LatentGreen(z_t,t)` 编码后作为额外条件 token 拼接

### 7.4 默认配置下的实际行为
当前 `default.yaml` 默认是：
- `cond_mode: concat`
- `use_green_attention: false`
- `condition_encoder.mode: map`

所以默认训练不会走 cross-attn，也不会在扩散模型内部调用 `LatentGreen` 做 attention 条件增强。

### 7.5 Patch 化与输出重建
`LatentDiffusion` 使用 `Conv2d(..., kernel_size=patch_size, stride=patch_size)` 做 patch embedding。
- 输入 latent 空间大小记为 `(H,W)`
- 输出 token 数为 `(H/patch_size) * (W/patch_size)`
- Transformer 输出后通过 `FinalLayer` 生成每个 patch 的像素向量
- `unpatchify()` 还原为 `(B, C_latent, H, W)`

### 7.6 结构细节（实现风格）
- 可学习位置编码 `pos_embed`
- Transformer depth 为 `depth`
- 前半段 block 的输出会缓存为 `skips`
- 后半段 block 会与对称位置 skip 做平均融合：`(x + skip) / 1.414`
- `AdaLN` 调制层和最终输出层初始被置零，输出初始更稳定（见 `initialize_weights()`）

### 7.7 预测目标类型
`prediction_type` 来自 `diffusion.training.prediction_type`，支持：
- `eps`：预测噪声
- `v`：预测 velocity（默认配置）
- `x0`：直接预测干净样本

`predict_eps()` 与 `predict_x0()` 会根据 `prediction_type` 做相互转换。

### 7.8 采样步 `step()` 的实现特征
`step(z_t, t, cond_input, eta)`：
- 先预测 `eps`
- 再估计 `x0`
- 构造 `z_prev = alpha_prev * x0 + sigma_prev * eps`
- 若 `eta > 0`，加随机噪声项

这是偏 DDIM 风格的单步更新实现（非标准 DDPM 后验公式全参数版）。

实现细节注意：
- `alpha_prev/sigma_prev` 使用 `(t - 1).clamp_min(1)`，因此不会显式走到 `t=0`。
- `TeacherSampler` 的时间序列是从 `T` 到 `1`，与上述实现保持一致。

### 7.9 噪声日程 `get_alpha_sigma()`
支持：
- `schedule="cosine"`
- 否则走线性形式 `alpha = 1 - t/T`

这里的 `alpha/sigma` 是当前项目自定义实现，不等同于所有论文中常见的 `alpha_bar` 参数化。

## 8. LatentDiffusion 的训练目标（`train_diffusion.py` 实际行为）
### 代码入口
- `gf/train/train_diffusion.py:29`

### 8.1 冻结依赖
训练 `LatentDiffusion` 时会冻结：
- `VAE`（用于 `V -> z`）
- `LatentGreen`（用于物理损失）

见 `gf/train/train_diffusion.py` 中 VAE 和 LatentGreen 初始化与 `requires_grad=False` 部分。

### 8.2 潜变量构造与缩放
训练脚本先做：
- `z, mu, _ = vae.encode(V)`
- 若 `mu is not None`，用 `mu` 替代采样 `z`（降低随机性，提高稳定性）

再按 `latent_scale` 配置处理：
- `auto`：按每样本标准差缩放到目标 `target_std`
- `fixed`：乘常数 `scale`
- `none`：不缩放

### 8.3 主扩散损失
训练随机采样 `t`，构造：
- `z_t = alpha_t * z + sigma_t * noise`

主损失根据 `prediction_type` 选择 target：
- `v`：`target = alpha_t * noise - sigma_t * z`
- `x0`：`target = z`
- `eps`：`target = noise`

支持 `min_snr` 加权（`train_diffusion.py` 中 `min_snr` 配置逻辑）：
- 使用 SNR 截断权重
- 针对 `v` 与非 `v` 模式使用不同分母

### 8.4 额外 `x0` 重建损失
当 `x0_loss_weight > 0` 时：
- 从 `pred` 反推出 `x0_pred`
- 计算 `MSE(x0_pred, z)`
- 按 `alpha_t^2` 做样本加权（减弱高噪声步的影响）

### 8.5 物理损失（通过 LatentGreen）
当 `phys_loss_weight > 0` 时：
- 使用 `x0_pred` 作为潜变量输入 `LatentGreen`
- 通常用 `t=0`（代码里用全零时间步）
- 得到 `g_pred_phys`
- 与 `g_obs` 做物理一致性损失

物理损失中包含的增强项（由训练脚本控制）：
- `phys_loss_type`：`mse/l1/huber`
- `per_energy_affine`
- `energy_align`（空间平移对齐）
- `energy_weights` 或动态 `snr/sensitivity` 能量权重
- `topk_phys`（仅对损失最大的 K 个能量通道取平均）
- `psd_loss_weight`
- `consistency_loss_weight`（相邻能量差分一致性）

物理损失同样按 `alpha_t^2` 做样本权重，并支持 warmup：
- `phys_warmup.start_ratio -> end_ratio`

### 8.6 训练工程特性
- 支持 AMP（fp16/bf16）
- 支持 DDP
- 支持 EMA（主进程维护和保存）
- 支持 cosine LR + warmup
- 周期性验证采样和可视化（调用 `TeacherSampler`）

## 9. StudentModel（学生网络）
### 代码定位
- 类定义：`gf/models/student.py:23`
- `forward`：`gf/models/student.py:40`

### 9.1 作用
直接学习从 `g_obs -> V` 的快速映射，用于蒸馏教师模型（TeacherSampler），实现更快推理。

### 9.2 结构
结构非常简单：
- `in_conv: Conv2d(K -> base_channels)`
- `num_res_blocks` 个 `ResBlock`
- `out_conv: Conv2d(base_channels -> 1)`

内部 `ResBlock` 与 VAE 风格类似，但定义独立存在于 `student.py`（未复用 `vae.py` 的类）。

### 9.3 输入输出
- 输入：`g_obs`，形状 `(B,K,H,W)`
- 输出：`V_stu`，形状 `(B,1,H,W)`

### 9.4 蒸馏训练（`train_student.py` 实际行为）
代码入口：`gf/train/train_student.py:23`

教师端：
- 用 `TeacherSampler` 生成 `V_teach`
- 教师内部会加载 `Diffusion + VAE + LatentGreen` checkpoint，并全部冻结

学生损失：
- 模仿损失 `imitation`：`L1` 或 `L2`
- 可选物理损失 `physics`：
  - `V_stu -> VAE.encode -> z_stu -> LatentGreen -> g_pred`
  - 与 `g_obs` 做 `mse / huber / charbonnier`
  - 物理项权重做 warmup

实现细节（建议知晓）：
- `train_student.py` 中学生物理损失使用 `vae.encode(V_stu)` 的返回 `z_stu`
- 若 `VAE.mode="vae"`，`encode` 默认会重参数采样，因此该物理损失会带随机性
- 与 `train_diffusion.py` 不同，学生训练脚本没有显式改用 `mu`

## 10. TeacherSampler（教师采样器）与推理流程
### 代码定位
- 类定义：`gf/inference/teacher_sampler.py:11`
- `sample`：`gf/inference/teacher_sampler.py:49`
- `_sample_steps`：`gf/inference/teacher_sampler.py:89`

### 10.1 作用
封装完整教师推理链：
- `LatentDiffusion` 采样潜变量
- 可选 `LatentGuidance` 物理引导
- 可选 `RestartSampler` 的 KPM 验证与重启
- `VAE.decode` 输出最终势场

### 10.2 采样流程
`sample(g_obs)` 的实际步骤：
1. 按配置生成随机潜变量 `z ~ N(0, I)`
2. 构造从 `T` 到 `1` 的离散时间步序列
3. 循环调用 `diffusion.step(z, t, g_obs, eta)`
4. 若启用 `guidance` 且满足时步条件，则调用 `LatentGuidance.apply`
5. 若启用 KPM 检查与重启，则执行若干轮 restart
6. 根据 `unscale_factor` 还原潜变量尺度
7. `vae.decode(z)` 得到 `V`

### 10.3 `unscale_factor` 的意义
当扩散训练使用 `latent_scale=auto/fixed` 时，验证或推理需要在 decode 前把潜变量尺度还原。
- `TeacherSampler.unscale_factor` 就是为此设计
- `train_diffusion.py` 的验证阶段会动态设置该值

### 10.4 KPM 校验与重启
如果 `validation.kpm_check.enabled` 且本机可导入 `kwant`：
- `TeacherSampler` 会创建 `RestartSampler`
- 每轮采样后可用 KPMForward 检查 `VAE.decode(z)` 的物理一致性
- 对不通过样本重新加噪并从较小 `t_restart` 再采样

若 `kwant` 缺失，`TeacherSampler` 会自动关闭该功能并打印 warning。

## 11. LatentGuidance（潜空间物理引导）
### 代码定位
- 类定义：`gf/guidance/latent_guidance.py:4`
- `apply`：`gf/guidance/latent_guidance.py:17`

### 11.1 作用
在扩散采样过程中，对潜变量 `z` 做一次可微物理修正：
- 使用 `LatentGreen(z, t)` 预测 `g_pred`
- 与 `g_obs` 计算一致性损失
- 对 `z` 求梯度并更新：`z <- z - lambda(t) * grad`

### 11.2 引导强度调度
`lambda(t)` 支持两种调度：
- `late_strong`：与 `(1 - alpha)^2` 相关，后期更强
- 其它模式：与 `sigma^2` 相关

在 `TeacherSampler` 中还叠加了一个时步门控：
- 仅当 `t <= guidance.lambda.start_step` 才启用引导（默认是后期引导）

## 12. RestartSampler（KPM校验与重启）
### 代码定位
- 类定义：`gf/guidance/restart.py:6`
- `check`：`gf/guidance/restart.py:18`
- `add_restart_noise`：`gf/guidance/restart.py:28`

### 12.1 作用
对采样结果做“物理真值检查”（用 KPMForward 而非 LatentGreen 代理）：
- `V_hat = vae.decode(z)`
- `g_pred = KPMForward(V_hat)`
- 计算 `delta = ||g_pred - g_obs||`
- 与阈值 `epsilon` 比较决定是否 restart

### 12.2 阈值策略
`epsilon` 支持：
- 绝对阈值
- 基于噪声估计（MAD 或 std）自适应阈值

### 12.3 重启噪声
`add_restart_noise` 会在 `t_restart` 的噪声水平重新扰动 `z`，再进入采样循环。

## 13. 默认配置（`default.yaml`）下的模型实例化画像
以下是对当前默认配置的“模型级别解读”（按代码读取路径）：

### 13.1 VAE
- `mode: vae`
- `latent_downsample: 2`
- `latent_channels: 8`
- `encoder.base_channels: 96`
- `encoder.num_res_blocks: 3`
- `recon_loss_type: log_cosh`
- `kl.weight: 1e-6`

结果含义：
- `64x64` 势场压缩为 `8x32x32` 潜变量
- 编码器较深，KL 极轻（更偏重重建）

### 13.2 LatentGreen
- `base_channels: 128`
- `num_res_blocks: 4`
- `use_timestep: true`
- `noisy_latent_training.enabled: true`
- 启用多种增强损失（FFT/PSD/stats/multiscale/residual 等）

结果含义：
- 该模型被设计成强物理代理，而不只是简单回归 `g`

### 13.3 Diffusion（LatentDiffusion）
- `T: 1000`
- `condition_encoder.mode: map`
- `cond_mode: concat`
- `patch_size: 2`
- `hidden_size: 384`
- `num_heads: 6`
- `depth: 12`
- `prediction_type: v`
- 启用 `min_snr`
- 启用 `ema`
- 启用 `phys_loss` 与 `x0_loss`

结果含义：
- 当前默认是“潜空间 DiT + concat 条件图 + 训练时物理约束”的方案
- `use_green_attention` 默认关闭，因此 `LatentGreen` 在 diffusion 里主要用于损失而不是条件注意力

### 13.4 Student
- `base_channels: 64`
- `num_res_blocks: 2`
- 启用 imitation loss 和 physics loss（含 warmup）

结果含义：
- 学生模型是一个轻量快速近似器，依赖教师质量上限

## 14. 模型兼容性与易错点（按当前实现）
### 14.1 条件编码模式与扩散条件模式要匹配
- `cond_mode="concat"` 时，`ConditionEncoder` 最好使用 `mode="map"`
- `cond_mode!=concat` 时，`ConditionEncoder` 应返回 token（`mode="token"` 或 `energy_seq"`）
- 若 cross-attn 路径错误地使用 `mode="map"`，张量维度会不匹配（4D map 无法直接喂给 `MultiheadAttention`）

### 14.2 cross-attn 的 token 维度必须等于 `hidden_size`
- 当前 `DiTBlock` 的 cross-attn 未设置 `kdim/vdim`
- 所以 `ConditionEncoder.token_dim == diffusion.model.hidden_size` 是必要条件（当使用 cross-attn 时）

### 14.3 VAE 解码器配置段未生效
- `vae.decoder.*` 在当前 `VAE` 构造函数中未使用
- 若你在配置里改 `vae.decoder.base_channels`，当前代码不会反映到模型结构

### 14.4 若启用 VAE 模式，`encode()` 默认含随机采样
- 在 `train_diffusion.py` 中已显式用 `mu` 替代采样 `z`
- 在 `train_student.py` 的物理损失路径中没有这一步，物理项会有随机噪声

### 14.5 `GroupNorm(8, channels)` 的隐含约束
- 相关模块里多处使用 `GroupNorm(8, channels)`
- 配置中的通道数应保持可被 8 整除，否则运行时报错

### 14.6 空间尺寸与 patch 约束
- `LatentDiffusion` 要求潜空间 `H/W` 可被 `patch_size` 整除
- 当前代码默认假设方形输入（`W = H`）

## 15. 按训练阶段理解“模型如何协同工作”
### 阶段1：VAE 训练
入口：`gf/train/train_vae.py:21`
- 训练 `VAE(V -> V_hat)`
- 保存 `vae_step_*.pt`
- 之后各阶段都依赖它

### 阶段2：LatentGreen 训练
入口：`gf/train/train_latent_green.py:23`
- 冻结 `VAE`
- `V -> z`（通常为 VAE 采样 latent）
- 可对 `z` 加噪训练
- 学习 `z -> g_obs`
- 叠加物理残差与频域损失
- 保存 `latent_green_step_*.pt`

### 阶段3：LatentDiffusion 训练
入口：`gf/train/train_diffusion.py:29`
- 冻结 `VAE` 与 `LatentGreen`
- `V -> z`
- 对 `z` 做扩散噪声训练
- 条件为 `g_obs`
- 主损失为扩散目标（`v/eps/x0`）
- 可叠加 `x0` 损失与 `LatentGreen` 物理损失
- 保存 `diffusion_step_*.pt` 与可选 `*_ema.pt`

### 阶段4：Student 蒸馏
入口：`gf/train/train_student.py:23`
- 冻结教师（Diffusion+VAE+LatentGreen）
- `TeacherSampler(g_obs) -> V_teach`
- 训练 `StudentModel(g_obs) -> V_stu`
- 用 imitation + physics loss 蒸馏
- 保存 `student_step_*.pt`

## 16. 最小使用范式（按当前代码风格）
### 16.1 VAE
```python
from gf.models.vae import VAE

vae = VAE(config).eval()
V = torch.randn(4, 1, 64, 64)
z, mu, logvar = vae.encode(V)
V_hat = vae.decode(z)
losses = vae.loss(V, V_hat, mu, logvar)
```

### 16.2 LatentGreen
```python
from gf.models.latent_green import LatentGreen

lg = LatentGreen(config).eval()
z = torch.randn(4, config["vae"]["latent_channels"], 32, 32)
t = torch.zeros(4, dtype=torch.long)
g_pred = lg(z, t)  # (B, K, 64, 64) in default config
```

### 16.3 LatentDiffusion（训练前向）
```python
from gf.models.diffusion import LatentDiffusion

model = LatentDiffusion(config)
z_t = torch.randn(4, config["vae"]["latent_channels"], 32, 32)
t = torch.randint(0, config["diffusion"]["T"], (4,))
g_obs = torch.randn(4, config["data"]["K"], 64, 64)
pred = model(z_t, t, g_obs)
```

### 16.4 Teacher 采样（完整推理）
```python
from gf.inference.teacher_sampler import TeacherSampler

teacher = TeacherSampler(config)
V_pred = teacher.sample(g_obs)  # (B, 1, H, W)
```

## 17. 测试与诊断脚本（推荐阅读顺序）
- `gf/test_vae.py`：VAE 重建测试与可视化
- `gf/test_dit.py`：LatentDiffusion 形状检查（包含 token_dim 与 hidden_size 的注意事项）
- `gf/test_latent_green.py`：LatentGreen 评估与多指标统计
- `gf/test_diffusion.py`：扩散采样诊断与可视化（含 EMA/归一化/物理误差检查）

这些脚本非常适合作为“文档配套示例”。

## 18. 一句话总结（按当前实现）
当前 GreenDiff 是一个“四模型协同”的体系：
- `VAE` 负责潜空间
- `LatentGreen` 负责可微物理代理
- `LatentDiffusion` 负责高质量条件生成
- `StudentModel` 负责快速蒸馏推理

并通过 `TeacherSampler + LatentGuidance + RestartSampler` 将生成先验与物理一致性在推理阶段闭环起来。
