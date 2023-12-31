from typing import Union, Tuple
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from lightlab.engine.onnx_predictor import OnnxPredictor
from lightlab.utils import LOGGER


def apply_coords(
    coords: np.ndarray, original_size: Tuple[int, ...], target_length
) -> np.ndarray:
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length
    )
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(
    boxes: np.ndarray, original_size: Tuple[int, ...], target_length
) -> np.ndarray:
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, target_length)
    return boxes.reshape(-1, 4)


def get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> Tuple[int, int]:
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


class Decoder(OnnxPredictor):
    img_size = (1024, 1024)
    mask_threshold = 0.0

    def __init__(self, model_path, device="cuda", warmup_epoch=1, **kwargs) -> None:
        super().__init__(model_path, 1024, 1, device, 0, **kwargs)

        self.imgsz = self.input_shape[2:]
        self.batch = self.input_shape[0]
        if warmup_epoch:
            self.warmup(warmup_epoch)

    def warmup(self, epoch: int) -> None:
        x = {
            "image_embeddings": np.random.random((1, 256, 64, 64)).astype(np.float32),
            "point_coords": np.random.random((1, 1, 2)).astype(np.float32),
            "point_labels": np.ones((1, 1), dtype=np.float32),
            "mask_input": np.random.random((1, 1, 256, 256)).astype(np.float32),
            "has_mask_input": np.ones((1,), dtype=np.float32),
            "orig_im_size": np.array((1024, 1024), dtype=np.float32),
        }
        LOGGER.info("start warmup!")
        for i in tqdm(range(epoch)):
            self.session.run(None, x)
        LOGGER.info("warmup finish!")

    def __call__(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        mask_input: np.ndarray = None,
    ) -> dict:
        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError(
                "Unable to segment, please input at least one box or point."
            )

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")
        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError("Got wrong mask!")
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = apply_coords(
                point_coords, origin_image_size, self.img_size[0]
            ).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4

            boxes = (
                apply_boxes(boxes, origin_image_size, self.img_size[0])
                .reshape((1, -1, 2))
                .astype(np.float32)
            )
            box_label = np.array(
                [[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32
            ).reshape((1, -1))

            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1

        input_dict = {
            "image_embeddings": img_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(origin_image_size, dtype=np.float32),
        }
        res = self.session.run(None, input_dict)

        result_dict = dict()
        for i in range(len(res)):
            out_name = self.session.get_outputs()[i].name
            if out_name == "masks":
                mask = (res[i] > self.mask_threshold).astype(np.int32)
                result_dict[out_name] = mask
            else:
                result_dict[out_name] = res[i]

        return result_dict
