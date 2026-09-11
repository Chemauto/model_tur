import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmb(nn.Module):
    """时间步 -> 正弦位置编码"""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim, self.max_period = dim, max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=1)


class ResBlock(nn.Module):
    """残差块, 融入时间嵌入"""

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """单头自注意力, 用在最低分辨率的瓶颈处"""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(h).chunk(3, dim=2)
        attn = (q @ k.transpose(1, 2)) * C ** -0.5
        h = attn.softmax(dim=-1) @ v
        return x + self.proj(h).transpose(1, 2).reshape(B, C, H, W)


class UNet(nn.Module):
    """噪声预测网络: 两次下采样, 跳跃连接, 瓶颈处自注意力"""

    def __init__(self, in_channels, base=64, mult=(1, 2, 4)):
        super().__init__()
        c1, c2, c3 = [base * m for m in mult]
        td = 128 * 4  # 时间嵌入维度
        self.time = nn.Sequential(SinusoidalEmb(128), nn.Linear(128, td), nn.SiLU(), nn.Linear(td, td))
        self.init_conv = nn.Conv2d(in_channels, c1, 3, 1, 1)
        self.d0a, self.d0b = ResBlock(c1, c1, td), ResBlock(c1, c2, td)
        self.down0 = nn.Conv2d(c2, c2, 3, 2, 1)
        self.d1a, self.d1b = ResBlock(c2, c2, td), ResBlock(c2, c3, td)
        self.down1 = nn.Conv2d(c3, c3, 3, 2, 1)
        self.mid1, self.attn, self.mid2 = ResBlock(c3, c3, td), Attention(c3), ResBlock(c3, c3, td)
        self.u0 = ResBlock(c3 * 2, c2, td)
        self.up0 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, 3, 1, 1))
        self.u1 = ResBlock(c2 * 2, c1, td)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c1, c1, 3, 1, 1))
        self.u2 = ResBlock(c1 * 2, c1, td)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, in_channels, 3, 1, 1))

    def forward(self, x, t):
        t_emb = self.time(t)
        s0 = self.init_conv(x)
        s1 = self.down0(self.d0b(self.d0a(s0, t_emb), t_emb))
        s2 = self.down1(self.d1b(self.d1a(s1, t_emb), t_emb))
        h = self.mid2(self.attn(self.mid1(s2, t_emb)), t_emb)
        h = self.u0(torch.cat([h, s2], 1), t_emb)
        h = self.up0(h)
        h = self.u1(torch.cat([h, s1], 1), t_emb)
        h = self.up1(h)
        h = self.u2(torch.cat([h, s0], 1), t_emb)
        return self.out(h)
