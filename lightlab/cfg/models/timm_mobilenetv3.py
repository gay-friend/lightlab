from dataclasses import dataclass
from timm.models.mobilenetv3 import default_cfgs

from . import MODELS
from lightlab.models.modules import ClassifyHead


@MODELS.register
@dataclass
class tf_mobilenetv3_large_075:
    backbone = [
        [
            -1,
            1,
            "tf_mobilenetv3_large_075",
            [default_cfgs["tf_mobilenetv3_large_075"].cfgs["in1k"].url],
        ],
    ]
    head = [
        [-1, 1, ClassifyHead, []],
    ]  # Classify
