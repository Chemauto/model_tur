import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


class EMA:
    """参数指数滑动平均: 采样时用 EMA 权重, 显著提升生成质量"""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


class DDPM(nn.Module):
    """去噪扩散概率模型
    前向: x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps,  训练目标是用 UNet 预测 eps
    采样: 从纯噪声出发, 按训练好的 eps 预测器逐步去噪"""

    def __init__(self, in_channels, base=64, mult=(1, 2, 4), timesteps=1000,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = timesteps
        self.unet = UNet(in_channels, base, mult)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        ab = torch.cumprod(1 - betas, dim=0)  # alpha_bar_t
        self.register_buffer("betas", betas)
        self.register_buffer("ab", ab)
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1ab", (1 - ab).sqrt())

    def losses(self, x, y=None):
        t = torch.randint(0, self.T, (x.size(0),), device=x.device)
        eps = torch.randn_like(x)
        xt = self.sqrt_ab[t].view(-1, 1, 1, 1) * x + self.sqrt_1ab[t].view(-1, 1, 1, 1) * eps
        loss = F.mse_loss(self.unet(xt, t), eps)
        return {"total": loss}

    def _x0_pred(self, xt, t, eps):
        a = self.sqrt_ab[t].view(-1, 1, 1, 1)
        b = self.sqrt_1ab[t].view(-1, 1, 1, 1)
        return (xt - b * eps) / a

    @torch.no_grad()
    def sample_ddpm(self, n, shape, device, record_ts=()):
        x = torch.randn(n, *shape, device=device)
        frames = []
        for step in reversed(range(self.T)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            eps = self.unet(x, t)
            mean = (x - self.betas[step] / self.sqrt_1ab[step] * eps) / (1 - self.betas[step]).sqrt()
            x = mean + self.betas[step].sqrt() * torch.randn_like(x) if step > 0 else mean
            if step in record_ts:
                frames.append(x.clone())
        return x, frames

    @torch.no_grad()
    def sample_ddim(self, n, shape, device, steps=50, record_ts=()):
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        x = torch.randn(n, *shape, device=device)
        frames = []
        for i, step in enumerate(ts):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            eps = self.unet(x, t)
            x0 = self._x0_pred(x, t, eps).clamp(-1, 1)
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps  # DDIM 确定性更新 (eta=0)
            if step in record_ts:
                frames.append(x.clone())
        return x, frames

    def sample(self, n, shape, device, sampler="ddim", ddim_steps=50, n_vis=8):
        record = set(torch.linspace(self.T - 1, 0, n_vis).round().long().tolist())
        if sampler == "ddim":
            return self.sample_ddim(n, shape, device, ddim_steps, record)
        return self.sample_ddpm(n, shape, device, record)
