import torch.nn as nn
import torch.nn.functional as F

from .blocks import Decoder, Encoder


class AE(nn.Module):
    """普通自编码器: 确定性映射, 没有隐空间概率结构, 因此无法直接采样生成"""

    def __init__(self, in_channels, img_size, latent_dim):
        super().__init__()
        feat_hw = img_size // 4
        self.encoder = Encoder(in_channels)
        self.fc = nn.Linear(128 * feat_hw * feat_hw, latent_dim)
        self.decoder = Decoder(in_channels, feat_hw, latent_dim)

    def encode(self, x):
        return self.fc(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def reconstruct(self, x, y=None):
        return self(x)

    def losses(self, x, y=None):
        recon = self(x)
        rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        return {"total": rec}
