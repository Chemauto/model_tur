# ⚙️ 配置

## 📄 config.yaml
唯一配置文件，所有实验只改它。

| 区块 | 作用 |
|---|---|
| 📊 `dataset` | 选数据集：`mnist` / `fashion` / `cifar10` |
| 📚 `datasets` | 各数据集定义（尺寸 / 通道 / 类别数） |
| 🎬 `run` | `mode`: train 或 infer；`model`: ae / vae / cvae / ddpm |
| 🏋️ `train` | 通用训练超参（batch / lr / epochs / 可视化频率） |
| 🎨 `infer` | 推理设置（样本数 / DDIM 步数 / CVAE 生成哪些数字） |
| 🧠 `model` | 各模型专属超参（每个模型可单独覆盖 epochs） |

## 🚀 用法
```bash
python main.py                      # 按默认配置训练
python main.py --mode infer         # 临时切推理（不改配置）
python main.py --digits 3,7         # CVAE 指定生成的数字
```
