# 🧠 模型

| 文件 | 作用 |
|---|---|
| 🏗️ `__init__.py` | `build_model(name, cfg)`：按 `run.model` 名字构建模型 |
| 🧱 `blocks.py` | AE / VAE / CVAE 共享的 Encoder 与 Decoder |
| 🔁 `ae.py` | AE：自编码器，只能重构，不能生成 |
| 🌊 `vae.py` | VAE：变分自编码器，可从 N(0,1) 采样生成 |
| 🏷️ `cvae.py` | CVAE：条件 VAE，指定数字生成对应图片 |
| 🌀 `ddpm.py` | DDPM：加噪训练 + DDPM/DDIM 两种采样 + EMA |
| 🕸️ `unet.py` | DDPM 的去噪网络（ResBlock + Attention + 时间嵌入） |

## 🔌 统一接口
所有模型实现 `losses(x, y=None)` 和 `reconstruct(x, y=None)`，训练与推理循环无需区分模型。
