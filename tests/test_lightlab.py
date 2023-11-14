from lightlab.models.segment import SegmentationModel
import torch
from torch import nn


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.relu(self.conv(x))


class LightLabDecoder(nn.Module):
    def __init__(self, imgsz=640, num_classes=2) -> None:
        super().__init__()
        self.feature6 = nn.Sequential(
            nn.Conv2d(120, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.feature7 = nn.Sequential(
            nn.Conv2d(48, 42, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)

        h, w = int(imgsz[0] / 32), int(imgsz[1] / 32)
        self.d5 = nn.Sequential(nn.Upsample(size=(h, w), mode="nearest"))
        self.d5_skip = SkipConnection(120, 42)

        h, w = h * 2, w * 2
        self.d4 = nn.Sequential(nn.Upsample(size=(h, w), mode="nearest"))
        self.d4_skip = SkipConnection(88, 42)

        h, w = h * 2, w * 2
        self.d3 = nn.Sequential(nn.Upsample(size=(h, w), mode="nearest"))
        self.d3_skip = SkipConnection(32, 42)

        h, w = h * 2, w * 2
        self.d2 = nn.Sequential(
            nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(
                42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Upsample(size=(h, w), mode="nearest"),
        )
        self.d2_skip = SkipConnection(24, 42)
        self.d2_d4_skip = nn.Sequential(
            nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(
                42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Upsample(size=(h, w), mode="bilinear"),
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(
                42, 42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
        )
        self.pre_out = nn.Sequential(
            nn.Conv2d(42, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=imgsz, mode="bicubic"),
        )

    def forward(self, *features):
        out = self.feature6(features[-1])
        print(f"feature6 {out.shape}, {features[-1].shape}")
        out = self.feature7(out)
        print(f"feature7 {out.shape}")
        out = self.d5(out)
        print(f"d5 {out.shape}")
        d5_skip = self.d5_skip(features[6])
        print(f"d5_skip {d5_skip.shape} {features[6].shape}")
        out = out + d5_skip

        d4_out = self.d4(out)
        print(f"d4 {d4_out.shape}")
        d4_skip = self.d4_skip(features[5])
        print(f"d4_skip {d4_skip.shape} {features[5].shape}")
        out = d4_out + d4_skip

        out = self.d3(out)
        print(f"d3 {out.shape}")
        d3_skip = self.d3_skip(features[3])
        print(f"d3_skip {d3_skip.shape} {features[3].shape}")
        out = out + d3_skip

        out = self.d2(out)
        print(f"d2 {out.shape}")
        d2_skip = self.d2_skip(features[2])
        print(f"d2_skip {d4_skip.shape} {features[2].shape}")
        out = out + d2_skip
        d2_d4_skip = self.d2_d4_skip(d4_out)
        print(f"d2_d4_skip {d2_d4_skip.shape} {d4_out.shape}")
        out = out + d2_d4_skip

        out = self.d1(out)
        print(f"d1{out.shape}")
        out = self.pre_out(out)
        print(f"pre_out{out.shape}")
        out = self.out(out)
        print(f"out{out.shape}")
        return out


import timm

model = timm.create_model(
    "tf_mobilenetv3_large_075",
    features_only=True,
    scriptable=True,
    exportable=True,
    # out_indices=(1, 2, 3, 4),
)
im = torch.zeros((3, 3, 640, 640))
# model._stage_out_idx = {s: i for i, s in enumerate([0, 1, 2, 3, 4, 5, 6])}
# features = model(im)
# for f in features:
#     print(f.shape)

# model = LightLabDecoder()
# preds = model(*features)
# torch.Size([3, 16, 320, 320])
# torch.Size([3, 16, 320, 320])
# torch.Size([3, 24, 160, 160])
# torch.Size([3, 32, 80, 80])
# torch.Size([3, 64, 40, 40])
# torch.Size([3, 88, 40, 40])
# torch.Size([3, 120, 20, 20])
# feature6 torch.Size([3, 48, 10, 10]), torch.Size([3, 120, 20, 20])
# feature7 torch.Size([3, 42, 5, 5])
# d5 torch.Size([3, 42, 20, 20])
# d5_skip torch.Size([3, 42, 20, 20]) torch.Size([3, 120, 20, 20])
# d4 torch.Size([3, 42, 40, 40])
# d4_skip torch.Size([3, 42, 40, 40]) torch.Size([3, 88, 40, 40])
# d3 torch.Size([3, 42, 80, 80])
# d3_skip torch.Size([3, 42, 80, 80]) torch.Size([3, 32, 80, 80])
# d2 torch.Size([3, 42, 160, 160])
# d2_skip torch.Size([3, 42, 40, 40]) torch.Size([3, 24, 160, 160])
# d2_d4_skip torch.Size([3, 42, 160, 160]) torch.Size([3, 42, 40, 40])
# d1torch.Size([3, 42, 160, 160])
# pre_outtorch.Size([3, 64, 160, 160])
# outtorch.Size([3, 2, 640, 640])

model = SegmentationModel("lightlab_mobilenetv3")
print(model)
pred = model(im)
print(pred.shape)
# 4,2,2,2,4,4
