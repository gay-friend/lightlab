from dataclasses import dataclass

from .models import (
    yolov8,
    yolov8_cls,
    timm_mobilenetv3,
    timm_efficientnet,
    lightlab_mobilenetv3,
)


@dataclass
class Settings:
    epochs: int = 500
    patience: int = 200
    batch: int = 8
    imgsz: int = 640
    save_period: int = -1
    cache: str = None
    device: str = "cuda"
    workers: int = 2
    save_dir: str = "output/tf_mobilenet"
    pretrained: int = True
    optimizer: str = "auto"
    verbose: bool = True
    seed: int = 0
    single_cls: bool = False
    amp: bool = False
    half: bool = False
    dynamic: bool = True
    simplify: bool = True
    opset = None
    overlap_mask: bool = True
    mask_ratio: int = 4
    dropout: float = 0.2  # classify train only
    conf: float = None
    iou: float = 0.7
    max_det: int = 300
    plots: bool = True
    rect: bool = False
    # Hyperparameters
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False  # (bool) use cosine learning rate scheduler
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    pose: float = 12.0
    kobj: float = 1.0
    label_smoothing: float = 0.0
    nbs: int = 64
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    degrees: float = 0.0
    perspective: float = 0.0  # image perspective (+/- fraction), range 0-0.001
    flipud: float = 0.0
    fliplr: float = 0.5
    copy_paste: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    close_mosaic: int = 10
    freeze = None
    pad: float = 0.5
    mask_ratio: float = 4.0
    overlap_mask: bool = True


def get_cfg(cfg: Settings, overrides: dict):
    if overrides is None:
        return cfg

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    cfg = Settings()
    print(cfg)
