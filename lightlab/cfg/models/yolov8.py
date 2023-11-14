from dataclasses import dataclass
from torch import nn

from . import MODELS
from lightlab.models.modules import Conv, C2f, SPPF, Concat, DetectHead, Conv, C2f


@MODELS.register
@dataclass
class yolov8:
    backbone = [
        [-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        [-1, 3, C2f, [128, True]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C2f, [256, True]],
        [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
        [-1, 6, C2f, [512, True]],
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C2f, [1024, True]],
        [-1, 1, SPPF, [1024, 5]],  # 9
    ]
    head = [
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C2f, [512]],  # 12
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C2f, [256]],  # 15 (P3/8-small)
        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 12], 1, Concat, [1]],  # cat head P4
        [-1, 3, C2f, [512]],  # 18 (P4/16-medium)
        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 9], 1, Concat, [1]],  # cat head P5
        [-1, 3, C2f, [1024]],  # 21 (P5/32-large)
        [[15, 18, 21], 1, DetectHead, []],  # Detect(P3, P4, P5)
    ]


if __name__ == "__main__":
    print(MODELS["yolov8"].head)
    print(MODELS["yolov8"].backbone)
