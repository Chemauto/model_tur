# ⚙️ 配置

## 📄 config.yaml
唯一配置文件，所有实验只改它。

| 区块 | 作用 |
|---|---|
| 📊 `dataset` | 选数据集：`mnist` / `fashion` / `cifar10` |
| 📚 `datasets` | 各数据集定义（尺寸 / 通道 / 类别数） |
| 🎬 `run` | `model`: ae / vae / cvae / ddpm，另有 seed 和 device |
| 🏋️ `train` | 通用训练超参（batch / lr / epochs / 可视化频率） |
| 🎨 `infer` | 推理设置（样本数 / DDIM 步数 / CVAE 生成哪些数字） |
| 🧠 `model` | 各模型专属超参（每个模型可单独覆盖 epochs） |

## 🚀 用法
```bash
python train.py                      # 训练
python infer.py                      # 推理
python infer.py --digits 3,7         # CVAE 指定生成的数字
python train.py --config 其他.yaml   # 使用其他配置文件
```
