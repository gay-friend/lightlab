from typing import Union

import numpy as np

from lightlab.engine.sam.vit import Vit
from lightlab.engine.sam.decoder import Decoder


class SamPredictor:
    def __init__(
        self, model_type="mobile", device: str = "cuda", warmup_epoch: int = 2, **kwargs
    ):
        vit_model_path = f"assets/sam/sam_{model_type}_encoder.onnx"
        decoder_model_path = f"assets/sam/sam_{model_type}.onnx"
        self.vit = Vit(vit_model_path, device, warmup_epoch, **kwargs)
        self.decoder = Decoder(decoder_model_path, device, warmup_epoch, **kwargs)

        self.features = None
        self.origin_image_size = None

    def register_image(self, img: np.ndarray) -> None:
        self.origin_image_size = img.shape
        # 重置特征图
        self.features = None

    def get_mask(
        self,
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        mask_input: Union[list, np.ndarray] = None,
    ) -> dict:
        # 获取特征图
        if self.features is None:
            start = time.time()
            self.features = self.vit(img)
            print("The encoder waist time: {:.3f}".format(time.time() - start))

        result = self.decoder(
            self.features,
            self.origin_image_size[:2],
            point_coords,
            point_labels,
            boxes,
            mask_input,
        )
        return result


if __name__ == "__main__":
    import cv2
    import time

    img = cv2.imread("assets/images/pole.jpg")
    points = []
    boxes = []
    box_point = []
    im0 = img.copy()
    get_first_box_point = False

    def change_box(box):
        x1 = min(box[0], box[2])
        y1 = min(box[1], box[3])
        x2 = max(box[0], box[2])
        y2 = max(box[1], box[3])
        return [x1, y1, x2, y2]

    def draw_circle(event, x, y, flags, param):
        global img, get_first_box_point, box_point
        if event == cv2.EVENT_LBUTTONDOWN and not get_first_box_point:
            print("Add point:", (x, y))
            points.append([x, y])

            mask = predictor.get_mask(
                points, [1 for i in range(len(points))], boxes if boxes != [] else None
            )

            mask = mask["masks"][0][0][:, :, None]
            cv2.imwrite("sam_mask.png", mask * 255)
            cover = np.ones_like(img) * 255
            cover = cover * mask
            cover = np.uint8(cover)
            img = cv2.addWeighted(im0, 0.6, cover, 0.4, 0)

            for b in boxes:
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
                cv2.circle(img, (b[0], b[1]), 5, (0, 0, 255), -1)
            for p in points:
                cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not get_first_box_point:
                box_point.append(x)
                box_point.append(y)
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                get_first_box_point = True
            else:
                box_point.append(x)
                box_point.append(y)
                box_point = change_box(box_point)
                boxes.append(box_point)
                print("add box:", box_point)

                mask = predictor.get_mask(
                    points if points != [] else None,
                    [1 for i in range(len(points))] if points != [] else None,
                    boxes,
                )

                mask = mask["masks"][0][0][:, :, None]
                cover = np.ones_like(img) * 255
                cover = cover * mask
                cover = np.uint8(cover)
                img = cv2.addWeighted(im0, 0.6, cover, 0.4, 0)

                get_first_box_point = False
                box_point = []

            for b in boxes:
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
                cv2.circle(img, (b[0], b[1]), 5, (0, 0, 255), -1)
                cv2.circle(img, (b[2], b[3]), 5, (0, 0, 255), -1)
            for p in points:
                cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)

        cv2.imshow("image", img)

    predictor = SamPredictor(device="cpu")

    predictor.register_image(img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle)
    key = cv2.waitKey()
    cv2.imwrite("sam.png", img)
    cv2.destroyAllWindows()
