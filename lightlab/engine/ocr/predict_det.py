import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

from lightlab.engine.onnx_predictor import OnnxPredictor


class PredictDet(OnnxPredictor):
    def __init__(
        self,
        model_path,
        imgsz=960,
        batch=1,
        device="cpu",
        warmup_epoch=1,
        box_thresh=0.6,
        unclip_ratio=1.6,
    ) -> None:
        super().__init__(model_path, imgsz, batch, device, warmup_epoch)
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

    def transform(self, img: np.ndarray) -> np.ndarray:
        img = super().transform(img, False, 32) / 255.0
        img = (img.astype(np.float32) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return img

    def postprocess(self, outputs: np.ndarray):
        thresh = 0.3
        max_candidates = 1000
        dilation_kernel = np.array([[1, 1], [1, 1]])
        min_size = 3

        pred = outputs[:, 0, :, :]
        segmentation = pred > thresh

        boxes_list = []
        for batch_index in range(pred.shape[0]):
            ori_h, ori_w = self.ori_shapes[batch_index][:2]
            mask = cv2.dilate(
                np.array(segmentation[batch_index]).astype(np.uint8),
                dilation_kernel,
            )
            h, w = mask.shape
            mask = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for contour in contours[:max_candidates]:
                points, sside = self.get_mini_boxes(contour)
                if sside < min_size:
                    continue

                points = np.array(points)
                score = self.box_score(pred[batch_index], points.reshape(-1, 2))

                if self.box_thresh > score:
                    continue
                box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
                box, sside = self.get_mini_boxes(box)
                if sside < min_size + 2:
                    continue
                box = np.array(box)

                box[:, 0] = np.clip(np.round(box[:, 0] / w * ori_w), 0, ori_w)
                box[:, 1] = np.clip(np.round(box[:, 1] / h * ori_h), 0, ori_h)
                boxes.append(box.astype(np.int16))
            boxes = np.array(boxes, dtype=np.int16)
            boxes = self.filter_tag_det_res(boxes, (ori_h, ori_w))
            boxes_list.append(boxes)
        return boxes_list

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def order_points_clockwise(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def box_score(self, mask, _box):
        h, w = mask.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        new_mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(new_mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(mask[ymin : ymax + 1, xmin : xmax + 1], new_mask)[0]

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])


if __name__ == "__main__":
    predictor = PredictDet("assets/ppocr/ppocrv4_det.onnx")
    out = predictor("assets/images/ppocr_det.png")
    print(out.shape)
    # import cv2
    # import numpy as np

    # cv2.imwrite("ppdet.png", (out[0][0] * 255).astype(np.uint8))
