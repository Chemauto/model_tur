import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Decoder, Encoder


class VAE(nn.Module):
    """变分自编码器: 编码为正态分布 N(mu, sigma^2), 重参数化采样后解码
    损失 = 重构误差 + KL(N(mu,sigma) || N(0,1)), KL 项把隐空间拉向标准正态,
    因此可以从 N(0,1) 直接采样生成"""

    def __init__(self, in_channels, img_size, latent_dim, kld_weight=1.0):
        super().__init__()
        feat_hw = img_size // 4
        self.kld_weight = kld_weight
        self.encoder = Encoder(in_channels)
        self.fc = nn.Linear(128 * feat_hw * feat_hw, 2 * latent_dim)  # mu 和 logvar
        self.decoder = Decoder(in_channels, feat_hw, latent_dim)

    def encode(self, x):
        mu, logvar = self.fc(self.encoder(x)).chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def reconstruct(self, x, y=None):
        return self(x)[0]

    def losses(self, x, y=None):
        recon, mu, logvar = self(x)
        rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return {"total": rec + self.kld_weight * kld, "recon": rec.detach(), "kld": kld.detach()}
