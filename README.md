# 🎨 model_tur

用 PyTorch 从零实现四种经典深度生成模型：**AE → VAE → CVAE → DDPM**，在统一的数据集和训练框架下对比它们的生成能力。

## 🧭 四个模型的故事线

| 模型 | 原理 | 能生成吗 | 预期现象 |
|------|------|----------|----------|
| 🔁 AE | 编码到确定向量，重构 | ✗ | 重构清晰，但隐空间无概率结构，随机采样得到噪声 |
| 🌊 VAE | 编码为正态分布 + KL 约束 | ✓（偏模糊） | 可从 N(0,1) 采样生成，隐空间支持平滑插值 |
| 🏷️ CVAE | VAE + 标签条件 | ✓（可控） | 指定数字生成对应图片，隐空间按类别分区块 |
| 🌀 DDPM | 逐步加噪 + UNet 学习去噪 | ✓（清晰） | 从纯噪声迭代去噪出清晰样本，质量最高 |

## 📁 目录结构

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

## 🏋️ 训练

**第 1 步：安装依赖**（只需一次）

```bash
pip install -r requirements.txt
```

**第 2 步：在 `configs/config.yaml` 里选数据集和模型**

```yaml
dataset: mnist        # 📊 可选: mnist / fashion (CPU 可训) / cifar10 (需 GPU)
run:
  model: cvae         # 🧠 可选: ae / vae / cvae / ddpm
```

**第 3 步：启动训练**

```bash
python train.py
```

**第 4 步：查看训练产物** `outputs/<模型>/<数据集>/`

- 💾 `model.pt` —— 模型权重（DDPM 额外含 EMA）
- 📉 `loss.png` —— 损失曲线
- 🖼️ `recon_ep*.png` / `preview_ep*.png` —— 训练过程可视化

> ⏱️ 换模型训练只需改 `run.model` 一个字段；换数据集只需改 `dataset` 一个字段。

## 🎨 推理

**第 1 步：确认有权重**

- 训练过的模型：`outputs/<模型>/<数据集>/model.pt` 已存在即可
- 💾 仓库自带已训练好的 CVAE 权重，克隆后无需训练即可体验

**第 2 步：把 `run.model` 改成要推理的模型**（须与权重对应的模型一致）

**第 3 步：运行推理**

```bash
python infer.py                # 🎲 通用推理: AE/VAE 展示采样与插值, DDPM 展示去噪过程
python infer.py --digits 3,7   # 🏷️ 仅 CVAE: 指定生成数字 3 和 7, 每个数字一行
python infer.py                # 🏷️ CVAE 不指定数字则 0-9 各生成一行
```

**第 4 步：查看推理产物**（保存在权重同目录）

| 模型 | 生成图片 | 内容 |
|------|----------|------|
| 🔁 AE | `ae_recon.png` / `ae_random.png` | 重构对比 / 随机采样失败现场 |
| 🌊 VAE | `vae_samples.png` / `vae_interp.png` | 随机采样 / 隐空间插值 |
| 🏷️ CVAE | `cvae_samples.png` | 按指定数字生成，一行一类 |
| 🌀 DDPM | `ddpm_samples.png` / `ddpm_steps.png` | 生成样本 / 去噪过程逐帧 |

## ⏱️ CPU 训练耗时参考（MNIST）

- 🔁 AE 约 2 分钟/epoch；🌊 VAE / 🏷️ CVAE 约 3 分钟/epoch；🌀 DDPM（base_channels=64）约 20 分钟/epoch
- 🐢 CPU 跑 DDPM 建议把 `model.ddpm.base_channels` 降到 `32`（效果略降，速度约 3 倍）
- ⚡ CIFAR-10 请使用 GPU（本机或 Colab），配置无需改动，换 `dataset: cifar10` 即可

## 🛠️ 常见问题

- 📥 **数据集下载慢/失败**：torchvision 需下载约 170MB，可手动下载放入 `data/` 对应目录（`data/` 不入 git）。
- 🌫️ **VAE 生成太模糊**：调小 `model.vae.kld_weight`（如 0.5），重构更清晰但采样分布会变差——这本身就是值得观察的权衡。
- 🏷️ **CVAE 生成混入其他数字**：类别信息从隐变量 z 泄漏是 CVAE 的经典现象。保持 `latent_dim: 16` 并使用已验证的 `kld_weight: 4.0` 即可干净生成；调参过程本身就是很好的实验题材。
- 🐌 **DDPM 采样太慢**：推理用默认的 `sampler: ddim`（50 步），比完整 1000 步 DDPM 快约 20 倍。
