from dataclasses import dataclass
from timm.models.mobilenetv3 import default_cfgs
from torch import nn

from . import MODELS
from lightlab.models.modules import SegmentationHead, LightConv, SkipConn


@MODELS.register
@dataclass
class lightlab_mobilenetv3:
    backbone = [
        [
            -1,
            1,
            "tf_mobilenetv3_large_075",
            # pretrain url, stages
            [default_cfgs["tf_mobilenetv3_large_075"].cfgs["in1k"].url, 1, 2, 3, 5, 6],
        ],  # 5 features of 16, 24, 32, 88, 120
    ]
    head = [
        [-1, 1, LightConv, [48, 3, 2, 1]],  # feature6 5
        [-1, 1, LightConv, [42, 3, 2, 1]],  # feature7 6
        [-1, 1, LightConv, [42, 3, 1, 1]],  # feature7 7
        [-1, 1, nn.Upsample, [None, 4, "nearest"]],  # d5 8
        [4, 1, SkipConn, [42]],  # d5 skip 9
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],  # d4 10
        [3, 1, SkipConn, [42]],  # d4 skip 11
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],  # d3 12
        [2, 1, SkipConn, [42]],  # d3 skip 13
        [-1, 1, LightConv, [42, 3, 1, 1, False, True]],  # d2 14
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],  # d2 15
        [1, 1, SkipConn, [42]],  # d2 skip
        [10, 1, SkipConn, [42, 3, 1, 1, False, True, 4, "bilinear"]],  # d2_d4_skip
        [-1, 1, LightConv, [42, 3, 1, 1, False, True]],  # d1
        [-1, 1, LightConv, [64, 3, 1, 1, False, True]],  # pre_out
        [-1, 1, SegmentationHead, [3, 1, 1, 4]],
    ]
