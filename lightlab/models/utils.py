import timm
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch

from lightlab.cfg.models import MODELS
from lightlab.utils import LOGGER
from lightlab.utils.torch_utils import make_divisible, intersect_dicts
from lightlab.models.modules import *


SCALES = dict(
    # [depth, width, max_channels]
    tiny=[0.33, 0.25, 1024],
    small=[0.33, 0.5, 1024],
    medium=[0.67, 0.75, 768],
    large=[1.00, 1.00, 512],
    extra_large=[1.00, 1.25, 512],
)


def parse_model(model_name="yolov8", ch=3, scale="tiny", nc=80, verbose=True):
    depth, width, max_channels = SCALES[scale]

    if verbose:
        print(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    model_cfg = MODELS[model_name]
    cfg = model_cfg.backbone + model_cfg.head
    is_backbone = False
    # from, number, module, args
    pre_len = 0
    for i, (f, n, m, args) in enumerate(cfg):
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (C2f, SPPF, Conv, LightConv, SkipConn):
            c1, c2 = ch[f], args[0]

            if (
                i + 1 < len(cfg)
                and cfg[i + 1][2] is not ClassifyHead
                and m not in (LightConv, SkipConn)
            ):
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m is C2f:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in (SegmentationHead, ClassifyHead):
            c1, c2 = ch[f], nc
            args = [c1, c2, *args]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is DetectHead:
            args = [nc]
            args.append([ch[x] for x in f])
        elif isinstance(m, str):
            t = m
            m: nn.Module = timm.create_model(
                m,
                pretrained=False,
                features_only=True,
                exportable=True,
                scriptable=True,
            )
            if args[0] is not None:
                state_dict = model_zoo.load_url(args[0])
                csd = m.float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, state_dict)  # intersect
                m.load_state_dict(csd, strict=False)  # load backbone

            if len(args[1:]) > 0:
                m._stage_out_idx = {s: i for i, s in enumerate(args[1:])}
                im = torch.zeros((1, 3, 128, 128))
                preds = m(im)
                c2 = [pred.shape[1] for pred in preds]
                pre_len = len(c2)
            else:
                c2 = m.feature_info.channels()
        else:
            c2 = ch[f]

        if isinstance(c2, list):
            is_backbone = True
            m_ = m
            m_.backbone = True
        else:
            # module
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")  # module type

        m.np = sum(x.numel() for x in m_.parameters())  # number params
        # attach index, 'from' index, type
        m_.i, m_.f, m_.type = (i + pre_len - 1 if is_backbone else i, f, t)
        if verbose:
            print(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")
        # append to savelist
        if is_backbone and isinstance(c2, list):
            save.extend([i + j for j in range(pre_len)])
        elif m is SkipConn:
            save.append(m_.i - 1)  # save pre for add
        else:
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

        layers.append(m_)
        if i == 0:
            ch = []

        if isinstance(c2, list):
            ch.extend(c2)
            for _ in range(5 - len(ch)):
                ch.insert(0, 0)
        else:
            ch.append(c2)
    return nn.Sequential(*layers), sorted(list(set(save)))
