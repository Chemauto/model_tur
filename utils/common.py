import random

import matplotlib
import torch
import torchvision

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_grid(x, path, nrow=8, title=""):
    """把一批 [-1,1] 的图像存成网格 PNG, 第一通道为 1 时按灰度显示"""
    grid = torchvision.utils.make_grid((x + 1) / 2, nrow=nrow).clamp(0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray" if x.size(1) == 1 else None)
    if title:
        plt.title(title, fontsize=9)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()


def plot_loss(history, path):
    plt.figure()
    plt.plot(history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
