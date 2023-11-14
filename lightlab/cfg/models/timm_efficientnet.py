from dataclasses import dataclass
from timm.models.efficientnet import default_cfgs

from . import MODELS
from lightlab.models.modules import ClassifyHead


@MODELS.register
@dataclass
class tf_efficientnet_b0:
    backbone = [
        [
            -1,
            1,
            "tf_efficientnet_b0",
            [default_cfgs["tf_efficientnet_b0"].cfgs["in1k"].url],
        ],
    ]
    head = [
        [-1, 1, ClassifyHead, []],
    ]  # Classify
