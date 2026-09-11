# 📦 输出

按 `outputs/<模型>/<数据集>/` 组织。

| 文件 | 说明 |
|---|---|
| 💾 `model.pt` | 权重（DDPM 额外含 EMA），推理时自动加载 |
| 📉 `loss.png` | 损失曲线 |
| 🏋️ `recon_ep*.png` / `preview_ep*.png` | 训练过程可视化 |
| 🎨 `ae_*.png` / `vae_*.png` / `cvae_*.png` / `ddpm_*.png` | 推理产物（重构 / 采样 / 插值 / 去噪过程） |

## 🚀 复现
```bash
python main.py                  # 训练
python main.py --mode infer     # 生成
```
