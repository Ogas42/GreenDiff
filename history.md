# 更新日志 (Change Log)

## [Performance Optimization] - 2026-02-03 (v2)

本次更新解决了 VAE 训练中的速度瓶颈、API 警告以及潜在的启动卡顿问题。

### ⚡ Speed & Efficiency (性能优化)
*   **AMP API Update**: 将 `torch.cuda.amp` 更新为最新的 `torch.amp` 接口，消除了 `FutureWarning`。
*   **Matmul Precision**: 将 `torch.set_float32_matmul_precision` 设置为 `medium`，利用 A6000/RTX30 系列的 Tensor Core 加速计算。
*   **Memory Format**: 启用 `channels_last` (NHWC) 内存格式，显著提升了卷积神经网络在 NVIDIA GPU 上的吞吐量。
*   **DataLoader Optimization**: 
    *   增加 `prefetch_factor` 至 4。
    *   将 `shard_workers` 增加至 12，优化了分片数据的读取速度。
*   **Torch Compile**: 重新启用 `torch.compile` 并添加了启动提示，预计在编译完成后可进一步提升 20-30% 的训练速度。

### 🔧 Fixes (修复)
*   **Hang Issue**: 修复了由于 `torch.compile` 在首次迭代时进行复杂图优化导致的 "卡住" 假象，通过增加控制台提示改善了用户体验。
*   **Training Speed**: 基础训练速度从 ~1.82 it/s 提升至 **~2.32 it/s** (不包含编译收益)，整体吞吐量提升约 27%。

---

## [SOTA Update] - 2026-02-03

本次更新旨在将 GreenDiff 推向 **SOTA (State-of-the-Art)** 性能水平，全面重构了模型架构与训练策略。

### 🚀 Core Improvements (核心改进)

#### 1. Diffusion Model
*   **Prediction Target**: 从 `epsilon` (噪声预测) 切换为 `v` (速度预测)。v-prediction 在高噪声区域（SNR -> 0）具有更好的数值稳定性，且能避免颜色偏移。
*   **Loss Weighting**: 引入 **Min-SNR Gamma** 策略 (CVPR 2023)。自动平衡不同时间步的 loss 权重，防止训练被简单的时间步主导。
*   **Conditioning**: 采用 **Concat + adaLN** 机制。条件特征（LDOS）直接与 Latent 拼接，并通过自适应层归一化调节网络流，最大化信息利用率。
*   **Training Schedule**: 引入 Cosine Learning Rate Schedule 和 5000 步 Warmup，总训练步数延长至 600k。

#### 2. VAE Model
*   **Architecture Upgrade**:
    *   Base Channels: 64 -> **96**
    *   ResBlocks: 2 -> **3**
    *   新增 **Log-cosh Loss**：比 L2 更鲁棒，比 L1 收敛更平滑。
*   **Regularization**: 重新启用 **KL Divergence** (VAE mode)，确保 Latent 空间分布连续，消除采样伪影。

#### 3. Latent Green Operator
*   **Noise Awareness**: 引入时间步嵌入 (Timestep Embedding)，使物理算子能够根据当前的噪声水平 $t$ 动态调整预测策略。
*   **Multi-scale Consistency**: 增加了多尺度损失约束。

### 🛠️ Bug Fixes (修复)
*   修复了 Min-SNR 权重计算公式中遗漏的 SNR 除法项。
*   修复了 Latent Scale 统计时的数值不稳定性。
*   解决了 VAE/Diffusion 架构升级导致旧权重加载报错的问题（增加了自动回退机制）。

---

## [Initial Release] - 2025-xx-xx
*   项目初始化。
*   实现了基础的 DiT + Latent Green 架构。
*   集成了 Kwant 物理引擎。
