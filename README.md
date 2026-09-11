# model_tur

用 PyTorch 从零实现四种经典深度生成模型：**AE → VAE → CVAE → DDPM**，在统一的数据集和训练框架下对比它们的生成能力。

## 四个模型的故事线

| 模型 | 原理 | 能生成吗 | 预期现象 |
|------|------|----------|----------|
| AE | 编码到确定向量，重构 | ✗ | 重构清晰，但隐空间无概率结构，随机采样得到噪声 |
| VAE | 编码为正态分布 + KL 约束 | ✓（偏模糊） | 可从 N(0,1) 采样生成，隐空间支持平滑插值 |
| CVAE | VAE + 标签条件 | ✓（可控） | 指定数字生成对应图片，隐空间按类别分区块 |
| DDPM | 逐步加噪 + UNet 学习去噪 | ✓（清晰） | 从纯噪声迭代去噪出清晰样本，质量最高 |

## 目录结构

```
configs/config.yaml   # 唯一配置文件: 数据集 / 模型 / 超参
train.py              # 训练入口
infer.py              # 推理入口
models/
  ae.py  vae.py  cvae.py   # 三者共享 blocks.py 中的 Encoder/Decoder
  ddpm.py  unet.py         # 扩散模型 (噪声调度 + DDPM/DDIM 采样) 与去噪 UNet
utils/
  data.py                  # MNIST / FashionMNIST / CIFAR10 加载
  common.py                # 种子 / 设备 / 存图
outputs/                   # 训练产物: outputs/<模型>/<数据集>/
```

## 快速开始

```bash
pip install -r requirements.txt
python train.py                # 按 config.yaml 默认设置训练 (mnist + cvae)
python infer.py --digits 3,7   # 推理: 指定生成数字 3 和 7
python infer.py                # 不指定则 10 类各生成一行
```

所有实验通过编辑 `configs/config.yaml` 驱动：

1. **选数据集**（顶部 `dataset:`）：`mnist` / `fashion`（CPU 可训）、`cifar10`（需 GPU）
2. **选模型**（`run.model:`）：`ae` / `vae` / `cvae` / `ddpm`

典型流程（以 DDPM 为例）：把 `run.model` 改为 `ddpm` 后

```bash
python train.py    # 训练
python infer.py    # 生成样本
```

仓库自带已训练好的 CVAE 权重（`outputs/cvae/mnist/model.pt`），克隆后可直接运行 `python infer.py --digits 3,7` 查看效果。

## 输出说明

训练产物在 `outputs/<模型>/<数据集>/`：`model.pt`（权重）、`loss.png`（损失曲线）、训练中的重构/预览图。

推理产物：

- **AE**：`ae_recon.png`（重构对比）、`ae_random.png`（随机采样失败现场）
- **VAE**：`vae_samples.png`（随机采样）、`vae_interp.png`（隐空间插值）
- **CVAE**：`cvae_samples.png`（按指定数字生成，一行一类）
- **DDPM**：`ddpm_samples.png`（生成样本）、`ddpm_steps.png`（去噪过程逐帧）

## CPU 训练耗时参考（MNIST）

AE 约 2 分钟/epoch，VAE / CVAE 约 3 分钟/epoch，DDPM（base_channels=64）约 20 分钟/epoch。
CPU 跑 DDPM 建议把 `model.ddpm.base_channels` 降到 `32`（效果略降，速度约 3 倍）。
CIFAR-10 请使用 GPU（本机或 Colab），配置无需改动，换 `dataset: cifar10` 即可。

## 常见问题

- **数据集下载慢/失败**：torchvision 需下载约 170MB，可手动下载放入 `data/` 对应目录（`data/` 不入 git）。
- **VAE 生成太模糊**：调小 `model.vae.kld_weight`（如 0.5），重构更清晰但采样分布会变差——这本身就是值得观察的权衡。
- **CVAE 生成混入其他数字**：类别信息从隐变量 z 泄漏是 CVAE 的经典现象。保持 `latent_dim: 16` 并使用已验证的 `kld_weight: 4.0` 即可干净生成；调参过程本身就是很好的实验题材。
- **DDPM 采样太慢**：推理用默认的 `sampler: ddim`（50 步），比完整 1000 步 DDPM 快约 20 倍。
