# 配置文件说明 (default.yaml)

本文档详细解释了 `gf/configs/default.yaml` 中的各项参数。该配置已更新为 **SOTA (State-of-the-Art)** 规格，旨在实现最高精度的物理反演。

## 1. Project Settings (项目基础)
```yaml
project:
  name: GreenDiff                # 项目名称
  seed: 42                       # 全局随机种子，保证可复现性
  device: cuda                   # 训练设备
  precision: fp16                # 混合精度训练 (推荐开启，节省显存)
```

## 2. VAE Settings (变分自编码器) - **Major Upgrade**
```yaml
vae:
  mode: vae                     # 模式: vae (推荐) | ae
  latent_channels: 8            # 隐空间通道数 (高容量)
  encoder:
    base_channels: 96           # 基础通道数 (从 64 提升至 96)
    num_res_blocks: 3           # ResBlock 堆叠数 (从 2 提升至 3)
  kl:
    weight: 1.0e-6              # KL 散度权重，用于正则化隐空间
  recon_loss_type: log_cosh     # 重建损失函数: log_cosh | l1 | mse
```
> **改动说明**: 架构大幅增强，引入 ResBlocks 和 Log-cosh Loss 以提升重建的锐利度和鲁棒性。

## 3. Diffusion Settings (扩散模型) - **Major Upgrade**
```yaml
diffusion:
  T: 1000                       # 总扩散步数
  schedule: cosine              # 噪声调度: cosine (推荐) | linear
  model:
    cond_mode: concat           # 条件注入: concat (强力注入) | cross_attn
    use_green_attention: false  # 是否融合 Latent Green 特征
  training:
    prediction_type: v          # 预测目标: v (velocity) - SOTA 标配
    min_snr:
      enabled: true             # 启用 Min-SNR Loss Weighting
      gamma: 5.0                # 权重截断阈值 (CVPR 2023)
    lr_schedule:
      type: cosine              # 学习率调度: cosine with warmup
      warmup_steps: 5000
    max_steps: 600000           # 训练总步数 (长时训练)
```
> **改动说明**: 采用了 v-prediction 和 Min-SNR 策略，这是目前训练高精度扩散模型的最佳实践。

## 4. Latent Green Settings (隐空间物理算子)
```yaml
latent_green:
  model:
    use_timestep: true          # 是否感知时间步 (Noise-aware)
  training:
    use_fft_loss: true          # 启用频域损失，捕捉周期性模式
    multiscale_loss_weight: 0.1 # 多尺度一致性约束
```

## 5. Guidance Settings (物理引导)
```yaml
guidance:
  lambda:
    lambda0: 8.0                # 引导强度基准
    schedule: late_strong       # 引导策略: 后期增强 (保护结构，修正细节)
    start_step: 400             # 介入时间点
```
