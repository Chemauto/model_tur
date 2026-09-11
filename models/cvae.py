import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Decoder, Encoder


class CVAE(nn.Module):
    """条件 VAE: 编码时把 one-hot 标签拼进卷积特征, 解码时拼进隐变量,
    使隐空间按类别分区块, 因此可以指定 '生成数字 k'
    损失与 VAE 相同: 重构误差 + KL 散度"""

    def __init__(self, in_channels, img_size, latent_dim, n_classes, kld_weight=1.0):
        super().__init__()
        feat_hw = img_size // 4
        self.latent_dim, self.n_classes, self.kld_weight = latent_dim, n_classes, kld_weight
        self.encoder = Encoder(in_channels)
        self.fc = nn.Linear(128 * feat_hw * feat_hw + n_classes, 2 * latent_dim)  # mu 和 logvar
        self.decoder = Decoder(in_channels, feat_hw, latent_dim + n_classes)

    def one_hot(self, y):
        return F.one_hot(y, self.n_classes).float()

    def encode(self, x, y):
        mu, logvar = self.fc(torch.cat([self.encoder(x), self.one_hot(y)], dim=1)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z, y):
        return self.decoder(torch.cat([z, self.one_hot(y)], dim=1))

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        return self.decode(self.reparameterize(mu, logvar), y), mu, logvar

    def reconstruct(self, x, y=None):
        return self(x, y)[0]

    @torch.no_grad()
    def generate(self, y):
        """按标签 y (LongTensor) 生成一批图像"""
        z = torch.randn(y.size(0), self.latent_dim, device=y.device)
        return self.decode(z, y)

    def losses(self, x, y=None):
        recon, mu, logvar = self(x, y)
        rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return {"total": rec + self.kld_weight * kld, "recon": rec.detach(), "kld": kld.detach()}
