import os
from glob import glob
import cv2
import numpy as np

# image suffixes
IMG_FORMATS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")

# global pin_memory for dataloaders
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"


def get_images(images_dir, suffixs=IMG_FORMATS, recursive=False):
    images = filter(
        lambda filename: filename.lower().endswith(suffixs),
        glob(f"{images_dir}/**", recursive=recursive),
    )
    return list(images)


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly first then resize is trying to keep the same way of loss calculation when mask-ratio=1.
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    return np.array(
        [
            polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio)
            for x in polygons
        ]
    )


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            imgsz,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index
