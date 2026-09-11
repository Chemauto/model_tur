import os

import torch
from tqdm import tqdm

from models import build_model
from models.ddpm import EMA
from utils.common import get_device, plot_loss, save_grid
from utils.data import get_dataloaders


@torch.no_grad()
def visualize(model, name, test_dl, device, out_dir, ep, ema=None):
    """AE/VAE 存重构对比图; DDPM 用 EMA 权重做轻量采样预览 (少量图片和步数)"""
    model.eval()
    x, y = next(iter(test_dl))
    x, y = x[:16].to(device), y[:16].to(device)
    if name == "ddpm":
        backup = {k: v.clone() for k, v in model.unet.state_dict().items()}
        if ema:
            model.unet.load_state_dict(ema.state_dict())
        samples, _ = model.sample(4, x.shape[1:], device, sampler="ddim", ddim_steps=16)
        save_grid(samples, os.path.join(out_dir, f"preview_ep{ep}.png"), nrow=4)
        model.unet.load_state_dict(backup)
    else:
        recon = model.reconstruct(x, y)
        save_grid(torch.cat([x, recon]), os.path.join(out_dir, f"recon_ep{ep}.png"), nrow=16,
                  title="top: real / bottom: recon")
    model.train()


def train(cfg):
    name = cfg["run"]["model"]
    spec = cfg["datasets"][cfg["dataset"]]
    tcfg = cfg["train"]
    device = get_device(cfg["run"]["device"])
    out_dir = os.path.join("outputs", name, cfg["dataset"])
    os.makedirs(out_dir, exist_ok=True)

    train_dl, test_dl = get_dataloaders(cfg)
    model, mcfg = build_model(name, cfg)
    model.to(device)
    epochs = mcfg.get("epochs", tcfg["epochs"])
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    ema = EMA(model.unet, cfg["model"]["ddpm"]["ema_decay"]) if name == "ddpm" else None

    print(f"model={name}  dataset={cfg['dataset']}  device={device}  epochs={epochs}")
    history = []
    for ep in range(1, epochs + 1):
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{epochs}")
        for step, (x, y) in enumerate(pbar):
            losses = model.losses(x.to(device), y.to(device))
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            if ema:
                ema.update(model.unet)
            if step % tcfg["log_every"] == 0:
                pbar.set_postfix({k: f"{v.item():.3f}" for k, v in losses.items()})
        history.append(losses["total"].item())
        if ep % tcfg["vis_every"] == 0 or ep == epochs:
            visualize(model, name, test_dl, device, out_dir, ep, ema)

    plot_loss(history, os.path.join(out_dir, "loss.png"))
    ckpt = {"model": model.state_dict(),
            "meta": {"model": name, "dataset": cfg["dataset"],
                     "in_channels": spec["channels"], "img_size": spec["img_size"]}}
    if ema:
        ckpt["ema"] = {f"unet.{k}": v for k, v in ema.state_dict().items()}
    torch.save(ckpt, os.path.join(out_dir, "model.pt"))
    print(f"training done. checkpoint -> {os.path.join(out_dir, 'model.pt')}")
