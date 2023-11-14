from dataclasses import dataclass
from torch import nn

from . import MODELS
from lightlab.models.modules import (
    Conv,
    C2f,
    ClassifyHead,
    Conv,
    C2f,
)


@MODELS.register
@dataclass
class yolov8_cls:
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
    ]
    head = [
        [-1, 1, Conv, [1280, 1, 1]],
        [-1, 1, ClassifyHead, []],
    ]  # Classify
