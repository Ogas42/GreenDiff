# GreenDiff 操作全流程指南 (Operational Workflow)

本文档提供从环境配置到模型最终推理的完整操作步骤。

> **注意**：本项目已升级至 **SOTA (State-of-the-Art)** 配置，对显存和算力有较高要求。默认配置针对 **NVIDIA RTX A6000 (48GB)** 级别的显卡优化。如果您使用消费级显卡（如 3090/4090），请适当降低 `batch_size`。

---

## 1. 环境准备 (Environment Setup)

建议使用 Anaconda 或 Miniconda 管理环境。

### 1.1 创建虚拟环境
```bash
conda create -n greendiff python=3.10
conda activate greendiff
```

### 1.2 安装依赖
```bash
# 1. 安装 PyTorch (建议 CUDA 11.8 或 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. 安装基础科学计算库
pip install numpy scipy pyyaml matplotlib tqdm

# 3. 安装 Kwant (用于物理数据生成与验证)
# Windows 下建议通过 conda 安装以避免编译问题
conda install -c kwant kwant
# 或者尝试 pip
pip install kwant
```

---

## 2. 核心配置说明 (Configuration)

核心配置文件位于 `gf/configs/default.yaml`。目前的配置为 **SOTA 高精度模式**。

*   **SOTA 配置 (默认)**:
    ```yaml
    vae:
      encoder:
        base_channels: 96       # 高容量 Encoder
        num_res_blocks: 3       # 深层 ResBlock
      training:
        max_steps: 200000       # 充分训练
    
    diffusion:
      training:
        prediction_type: v      # v-prediction 模式
        min_snr:
          enabled: true         # 开启 Min-SNR 加速
        max_steps: 600000       # 长时训练
    ```

*   **消费级显卡优化建议 (如 8GB-24GB 显存)**:
    如果您显存不足，请在 `default.yaml` 中修改以下参数：
    ```yaml
    project:
      precision: fp16         # 务必开启混合精度
    vae:
      training:
        batch_size: 32        # 降低 Batch Size (默认 256)
    diffusion:
      training:
        batch_size: 16        # 降低 Batch Size (默认 256)
    ```

---

## 3. 全流程操作步骤 (Pipeline)

### Step 1: 物理数据生成 (Data Generation)
利用 Kwant 生成训练所需的势能 $V$ 和对应的 $LDOS$ 数据。

```bash
# 运行此命令会自动在 data_cache 目录下生成 .pt 数据文件
python -m gf.data.dataset
```
*   **耗时预估**: 视样本量而定。

### Step 2: 训练 VAE (Stage 2)
训练高保真变分自编码器。**注意：由于架构升级，旧权重的加载会自动被跳过，训练将从零开始。**

```bash
python -m gf.train.train_vae
```
*   **SOTA 目标**: 200,000 Steps
*   **作用**: 为扩散模型提供高质量的“数字底片”。

### Step 3: 训练隐空间格林算子 (Stage 3)
训练一个能够感知噪声强度的物理算子网络。

```bash
python -m gf.train.train_latent_green
```
*   **SOTA 目标**: 40,000 Steps
*   **作用**: 提供多尺度的物理梯度引导。

### Step 4: 训练 DiT 扩散模型 (Stage 4 - Core)
训练核心的 Diffusion Transformer。

```bash
python -m gf.train.train_diffusion
```
*   **SOTA 目标**: 600,000 Steps
*   **特性**: 训练初期可能会看到 Loss 较高，这是 v-prediction 的特性，随着 Cosine Schedule 的推进，Loss 会稳步下降。

---

## 4. 常见问题 (FAQ)

**Q: 为什么 VAE 训练一开始 Loss 很大？**
A: 因为我们使用了 Log-cosh Loss 和 KL 散度约束，且从零开始训练。请耐心等待几千步，模型会迅速收敛。

**Q: Diffusion 训练时的 SNR 和 Weight 统计信息是什么意思？**
A: 这是 Min-SNR 策略的监控指标。`w_mean` 反映了当前 batch 的平均损失权重。如果 `z_std` 稳定在 1.0 附近，说明 Latent Scale 正常。
