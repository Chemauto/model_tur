import torch.nn as nn


class Encoder(nn.Module):
    """(C, S, S) -> 两次 stride-2 卷积下采样 -> (128, S/4, S/4) -> flatten"""

    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),   # S -> S/2
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),            # S/2 -> S/4
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class Decoder(nn.Module):
    """in_dim 维向量(隐变量, 可再拼接条件标签) -> 反卷积上采样回原尺寸, 输出经 tanh 映射到 [-1, 1]"""

    def __init__(self, out_channels, feat_hw, in_dim):
        super().__init__()
        self.feat_hw = feat_hw
        self.fc = nn.Linear(in_dim, 128 * feat_hw * feat_hw)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # S/4 -> S/2
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # S/2 -> S
            nn.Conv2d(32, out_channels, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, self.feat_hw, self.feat_hw)
        return self.net(x)
