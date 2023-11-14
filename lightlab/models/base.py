import torch
import torch.nn as nn
import thop


from .utils import parse_model
from lightlab.utils.torch_utils import (
    intersect_dicts,
    time_sync,
    fuse_conv_and_bn,
    model_info,
)
from lightlab.utils import LOGGER
from lightlab.utils.plotting import feature_visualization
from .modules import Conv, DetectHead, LightConv, SkipConn


class BaseModel(nn.Module):
    def __init__(
        self,
        model_name="yolov8_cls",
        scale="tiny",
        ch=3,
        nc=80,
        names=None,
        verbose=True,
    ):
        super().__init__()
        # model, savelist
        self.model, self.save = parse_model(
            model_name, scale=scale, ch=ch, verbose=verbose, nc=nc
        )
        self.stride = torch.Tensor([1])  # no stride constraints
        # default names dict
        self.names = {i: f"{i}" for i in range(nc)} if names is None else names
        if verbose:
            self.info()

    def forward(self, x, profile=False, visualize=False, augment=False):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize)

    def _predict_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if hasattr(m, "backbone"):
                x = m(x)
                for _ in range(5 - len(x)):
                    x.insert(0, None)
                for i_idx, i in enumerate(x):
                    if i_idx in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                x = x[-1]
            else:
                x = m(x)  # run
                if isinstance(m, SkipConn):
                    x = x + y[-1]
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. "
            f"Reverting to single-scale inference instead."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        # is final layer list, copy input as inplace fix
        c = m == self.model[-1] and isinstance(x, list)
        flops = (
            thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2
            if thop
            else 0
        )  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        if not self.is_fused():
            for m in self.model.modules():
                if (
                    isinstance(m, (Conv, SkipConn, LightConv))
                    and hasattr(m, "bn")
                    and not isinstance(m.bn, nn.Identity)
                ):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        # normalization layers, i.e. BatchNorm2d()
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        # True if < 'thresh' BatchNorm layers in model
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, DetectHead):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        # checkpoint state_dict as FP32
        csd = torch.load(weights)
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)
        if verbose:
            LOGGER.info(
                f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights"
            )
