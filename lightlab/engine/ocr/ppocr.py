import numpy as np
from pathlib import Path
import cv2
from copy import deepcopy

from lightlab.engine.ocr.predict_det import PredictDet
from lightlab.engine.ocr.predict_cls import PredictCls
from lightlab.engine.ocr.predict_rec import PredictRec


class PPOcr:
    def __init__(
        self,
        use_angle_cls=True,
        box_thresh=0.5,
        unclip_ratio=2,
        device="cpu",
    ) -> None:
        self.use_angle_cls = use_angle_cls
        self.text_detector = PredictDet(
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            device=device,
            model_path="assets/ppocr/ppocrv4_det.onnx",
        )
        self.text_recognizer = PredictRec(
            model_path="assets/ppocr/ppocrv4_rec.onnx",
            device=device,
            imgsz=[48, 320],
        )
        if use_angle_cls:
            self.text_classifier = PredictCls(
                model_path="assets/ppocr/ppocrv2_cls.onnx",
                imgsz=[48, 192],
                device=device,
            )

    def __call__(self, img: np.ndarray, drop_score=0.5):
        if isinstance(img, (str, Path)):
            img = cv2.imread(Path(img).as_posix())
        ori_im = img.copy()
        dt_boxes = self.text_detector(img)[0]
        if dt_boxes is None:
            return []
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for box in dt_boxes:
            tmp_box = deepcopy(box)
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
            # import time

            # cv2.imwrite(f"{time.time()}.png", img_crop)

        if self.use_angle_cls:
            rotates = self.text_classifier(img_crop_list)
            for i, (rotate, conf) in enumerate(rotates):
                if conf > self.text_classifier.cls_thresh and "180" in rotate:
                    img_crop_list[i] = cv2.rotate(img_crop_list[i], 1)
        rec_res = self.text_recognizer(img_crop_list)
        print(rec_res)


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (
            _boxes[i + 1][0][0] < _boxes[i][0][0]
        ):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


if __name__ == "__main__":
    ocr = PPOcr()
    ocr("assets/images/ppocr_det.png")
