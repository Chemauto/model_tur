import os

import torch

from utils.common import get_device, save_grid, set_seed
from utils.data import get_dataloaders


@torch.no_grad()
def infer(cfg):
    set_seed(cfg["run"]["seed"])
    name = cfg["run"]["model"]
    icfg = cfg["infer"]
    device = get_device(cfg["run"]["device"])
    path = icfg["checkpoint"] or os.path.join("outputs", name, cfg["dataset"], "model.pt")
    out_dir = os.path.dirname(path)

    from models import build_model
    model, _ = build_model(name, cfg)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if name == "ddpm" and "ema" in ckpt:  # EMA 只含 UNet 参数, 叠加在完整权重之上
        sd = model.state_dict()
        sd.update(ckpt["ema"])
        model.load_state_dict(sd)
    model.to(device).eval()
    print(f"loaded checkpoint: {path}  (trained on {ckpt['meta']['dataset']})")

    x, y = next(iter(get_dataloaders(cfg)[1]))
    x, y = x[:16].to(device), y[:16].to(device)

    if name == "ae":
        save_grid(torch.cat([x, model.reconstruct(x)]), os.path.join(out_dir, "ae_recon.png"),
                  nrow=16, title="AE: top=real / bottom=recon")
        # AE 没有概率结构: 从 N(0,1) 直接采样解码, 结果是噪声 -> 引出 VAE
        z = torch.randn(16, cfg["model"]["ae"]["latent_dim"], device=device)
        save_grid(model.decode(z), os.path.join(out_dir, "ae_random.png"), nrow=16,
                  title="AE: decode N(0,1) samples (fails to generate)")
        print("结论: AE 隐空间无概率结构, 随机采样无法生成图像")
    elif name == "vae":
        z = torch.randn(16, cfg["model"]["vae"]["latent_dim"], device=device)
        save_grid(model.decode(z), os.path.join(out_dir, "vae_samples.png"), nrow=16,
                  title="VAE: N(0,1) random samples")
        # 隐空间插值: 两张测试图之间平滑过渡, 体现隐空间的连续性
        mu_a, _ = model.encode(x[0:1])
        mu_b, _ = model.encode(x[1:2])
        alphas = torch.linspace(0, 1, 10, device=device).view(-1, 1)
        save_grid(model.decode(mu_a * (1 - alphas) + mu_b * alphas),
                  os.path.join(out_dir, "vae_interp.png"), nrow=10, title="VAE: latent interpolation")
        print("结论: VAE 可从 N(0,1) 采样生成 (但偏模糊), 且隐空间支持插值")
    elif name == "cvae":
        n = icfg["n_samples"]
        digits = icfg.get("digits") or list(range(cfg["datasets"][cfg["dataset"]]["classes"]))
        rows = [model.generate(torch.full((n,), d, dtype=torch.long, device=device)) for d in digits]
        save_grid(torch.cat(rows), os.path.join(out_dir, "cvae_samples.png"), nrow=n,
                  title=f"CVAE: one row per digit {digits}")
        print(f"结论: CVAE 可指定类别生成, 每个数字一行 -> cvae_samples.png")
    else:
        n = icfg["n_samples"]
        samples, frames = model.sample(n, x.shape[1:], device, icfg["sampler"],
                                       icfg["ddim_steps"], icfg["n_vis_steps"])
        save_grid(samples, os.path.join(out_dir, "ddpm_samples.png"), nrow=n,
                  title=f"DDPM ({icfg['sampler']}) samples")
        save_grid(torch.cat(frames), os.path.join(out_dir, "ddpm_steps.png"), nrow=n,
                  title="DDPM: denoising process, one row per timestep")
        print("结论: DDPM 迭代去噪生成, 质量明显高于 VAE")

    print(f"results saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="推理 (模型/数据集在 config.yaml 中选择)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--digits", type=str, help="CVAE 指定生成的数字, 逗号分隔, 如 3,7")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.digits:
        cfg["infer"]["digits"] = [int(d) for d in args.digits.split(",")]
    infer(cfg)
