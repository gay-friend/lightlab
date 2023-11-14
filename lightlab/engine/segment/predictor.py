import numpy as np
import cv2


from lightlab.engine.onnx_predictor import OnnxPredictor


class Predictor(OnnxPredictor):
    def postprocess(self, outputs: np.ndarray):
        masks = outputs.argmax(axis=1).astype(np.uint8)
        shapes_list = []
        for mask in masks:
            cv2.imwrite("mask.png", mask * 255)
            shapes = []
            for i in set(mask.reshape(-1).tolist()):
                contours, _ = cv2.findContours(
                    np.where(mask == i, mask, 0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = filter(
                    lambda contour: cv2.contourArea(contour) >= 5, contours
                )
                contours = [
                    contour.squeeze(axis=1).astype(np.float32) for contour in contours
                ]
                shapes.append(contours)
                # for contour in contours:
                #     points = contour.squeeze(axis=1).astype(np.float32)
                #     print(points.shape)
                # points[:, 0] *= scale_w
                # points[:, 1] *= scale_h
            shapes_list.append(shapes)
        return shapes_list

    def preprocess(self, img: np.ndarray):
        im = super().preprocess(img) / 255.0
        return im


if __name__ == "__main__":
    model = Predictor("output/segment/weights/best.onnx")
    file = (
        "datasets/bushing_sleeve_od_abration/val/images/1614388947416-WE74Q8U7EJ-0.png"
    )
    model(file)
