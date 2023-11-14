from torch import nn
import torch


def autopad(k, p=None, d=1):
    # kernel padding dilation
    if d > 1:
        k = (
            d * (k - 1) if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class LightConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        k=3,
        s=2,
        p=1,
        bias=True,
        bn=False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.relu(self.conv(x))


class SkipConn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        k=3,
        s=1,
        p=1,
        bias=True,
        bn=False,
        upsampling=1,
        mode="bilinear",
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.LeakyReLU()
        self.up = (
            nn.Upsample(scale_factor=upsampling, mode=mode)
            if upsampling > 1
            else nn.Identity()
        )

    def forward(self, x):
        return self.up(self.relu(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.up(self.relu(self.conv(x)))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
