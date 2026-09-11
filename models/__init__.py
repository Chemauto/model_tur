from .ae import AE
from .cvae import CVAE
from .ddpm import DDPM
from .vae import VAE


def build_model(name, cfg):
    """按 run.model 的名字构建模型, 返回 (model, 剩余超参字典)"""
    spec = cfg["datasets"][cfg["dataset"]]
    m = dict(cfg["model"][name])
    if name == "ae":
        model = AE(spec["channels"], spec["img_size"], m.pop("latent_dim"))
    elif name == "vae":
        model = VAE(spec["channels"], spec["img_size"], m.pop("latent_dim"), m.pop("kld_weight", 1.0))
    elif name == "cvae":
        model = CVAE(spec["channels"], spec["img_size"], m.pop("latent_dim"),
                     spec["classes"], m.pop("kld_weight", 1.0))
    elif name == "ddpm":
        model = DDPM(spec["channels"], m.pop("base_channels", 64), m.pop("mult", [1, 2, 4]),
                     m.pop("timesteps"), m.pop("beta_start"), m.pop("beta_end"))
    else:
        raise ValueError(f"unknown model: {name}")
    return model, m
