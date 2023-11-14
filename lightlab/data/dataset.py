from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import math
import cv2
import os
import torch

from lightlab.data.augment import (
    classify_transforms,
    v8_transforms,
    segment_transforms,
    Format,
    Compose,
    LetterBox,
)
from lightlab.cfg import Settings
from lightlab.data.utils import get_images
from lightlab.utils.instance import Instances
from lightlab.utils.ops import segments2boxes


class ClassificationDataset(ImageFolder):
    def __init__(self, root, args: Settings = Settings(), augment=False, cache=False):
        super().__init__(root=root)
        images = get_images(images_dir=root, recursive=True)
        self.names = []
        self.samples = []
        for im_file in images:
            name = Path(im_file).parent.name
            if name not in self.names:
                self.names.append(name)
            self.samples.append((im_file, self.names.index(name)))

        self.transforms = classify_transforms(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225],
        )

    def __getitem__(self, index):
        im_file, cls_index = self.samples[index]
        im = Image.open(im_file).convert("RGB")
        im = self.transforms(image=np.array(im))["image"]
        return {"img": im, "cls": cls_index}

    def __len__(self):
        return len(self.samples)


class YoloDataset(Dataset):
    labels_name = "labels"
    labels_suffix = ".txt"

    def __init__(
        self,
        root,
        cfg=Settings(),
        augment=True,
        use_segments=False,
        use_keypoints=False,
        kpt_shape=(0, 0),
        flip_idx=[],
        stride=32,
        rect=False,
    ):
        super().__init__()
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.kpt_shape = kpt_shape
        self.flip_idx = flip_idx
        assert not (
            self.use_segments and self.use_keypoints
        ), "Can not use both segments and keypoints."

        self.root = Path(root)
        self.imgsz = cfg.imgsz
        self.stride = stride
        self.augment = augment
        self.single_cls = cfg.single_cls
        self.rect = rect
        self.batch_size = cfg.batch
        self.pad = cfg.pad
        self.cfg = cfg

        images_dir = self.root / "images"
        labels_dir = self.root / self.labels_name
        images = get_images(images_dir=images_dir, recursive=True)
        self.samples = [
            (file, labels_dir.joinpath(Path(file).name).with_suffix(self.labels_suffix))
            for file in images
        ]

        self.ni = len(self.samples)

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = (
            min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        )

        self.ims, self.im_hw0, self.im_hw = (
            [None] * self.ni,
            [None] * self.ni,
            [None] * self.ni,
        )

        # Transforms
        self.transforms = self.build_transforms()

    def __len__(self):
        return len(self.samples)

    def load_image(self, i):
        im = self.ims[i]
        im_file, _ = self.samples[i]

        if im is not None:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]

        im = cv2.imread(im_file)  # BGR
        if im is None:
            raise FileNotFoundError(f"Image Not Found {im_file}")

        h0, w0 = im.shape[:2]  # orig hw
        r = self.imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            w, h = (
                min(math.ceil(w0 * r), self.imgsz),
                min(math.ceil(h0 * r), self.imgsz),
            )
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add to buffer if training with augmentations
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                im,
                (h0, w0),
                im.shape[:2],
            )  # im, hw_original, hw_resized
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index))

    def get_label(self, label_file):
        nkpt, ndim = self.kpt_shape
        lb = np.zeros(
            (0, (5 + nkpt * ndim) if self.use_keypoints else 5), dtype=np.float32
        )
        segments, keypoints = [], None

        if not os.path.isfile(label_file):
            return lb, segments, keypoints

        lb = [
            x.split()
            for x in Path(label_file).read_text().strip().splitlines()
            if len(x)
        ]
        if any(len(x) > 6 for x in lb) and (not self.use_keypoints):  # is segment
            classes = np.array([x[0] for x in lb], dtype=np.float32)
            # (cls, xy1...)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
            # (cls, xywh)
            lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
        lb = np.array(lb, dtype=np.float32)

        nl = len(lb)
        if nl:
            if self.use_keypoints:
                assert lb.shape[1] == (
                    5 + nkpt * ndim
                ), f"labels require {(5 + nkpt * ndim)} columns each"
                points = lb[:, 5:].reshape(-1, ndim)[:, :2]
            else:
                assert (
                    lb.shape[1] == 5
                ), f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
            assert (
                points.max() <= 1
            ), f"non-normalized or out of bounds coordinates {points[points > 1]}"
            assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

            _, i = np.unique(lb, axis=0, return_index=True)
            if len(i) < nl:  # duplicate row check
                lb = lb[i]  # remove duplicates
                if segments:
                    segments = [segments[x] for x in i]
        return lb, segments, keypoints

    def get_image_and_label(self, index):
        im_file, label_file = self.samples[index]
        lb, segments, keypoints = self.get_label(label_file)

        img, (h0, w0), (h, w) = self.load_image(index)
        ratio_pad = (h / h0, w / w0)

        return dict(
            im_file=im_file,
            img=img,
            ori_shape=(h0, w0),
            resized_shape=(h, w),
            ratio_pad=ratio_pad,
            cls=lb[:, 0:1],
            instances=Instances(
                bboxes=lb[:, 1:],
                segments=segments,
                keypoints=keypoints,
                bbox_format="xywh",
                normalized=True,
            ),
        )

    def build_transforms(self):
        if self.augment:
            self.cfg.mosaic = self.cfg.mosaic if self.augment and not self.rect else 0.0
            self.cfg.mixup = self.cfg.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, self.cfg)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=self.cfg.mask_ratio,
                mask_overlap=self.cfg.overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self):
        self.cfg.mosaic = 0.0  # set mosaic ratio=0.0
        self.cfg.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.cfg.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms()

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class SegmentDataset(Dataset):
    def __init__(self, root="", imgsz=640, augment=True) -> None:
        super().__init__()

        self.root = Path(root)
        self.images_dir = self.root.joinpath("images")
        self.labels_dir = self.root.joinpath("masks")
        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        else:
            self.imgsz = imgsz

        images = get_images(self.images_dir)
        self.samples = []
        for im_file in images:
            mask_file = self.labels_dir.joinpath(f"{Path(im_file).stem}.png")
            if mask_file.is_file():
                self.samples.append((im_file, mask_file.as_posix()))

        self.transforms = segment_transforms(augment, imgsz=self.imgsz)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        im_file, label_file = self.samples[index]
        im = Image.open(im_file).convert("RGB")
        mask = Image.open(label_file)
        sample = self.transforms(image=np.array(im), mask=np.array(mask))
        return dict(im_file=im_file, img=sample["image"], mask=sample["mask"])


if __name__ == "__main__":
    dataset = YoloDataset("datasets/coco128/train")
    import time

    batch = dataset[0]
    batch = dataset[1]
    batch = dataset[2]
    batch = dataset[4]
    for i in range(60):
        batch = dataset[4]
        img: np.ndarray = batch["img"].numpy()
        img = img.transpose((1, 2, 0))
        im = Image.fromarray(img)
        im.save("output/test.png")
        time.sleep(1)
