import numpy as np


from lightlab.engine.onnx_predictor import OnnxPredictor


class Predictor(OnnxPredictor):
    def postprocess(self, outputs: np.ndarray):
        preds = outputs.argmax(axis=1)
        print(preds)
        return preds

    def preprocess(self, img: np.ndarray):
        im = super().preprocess(img) / 255.0
        return im


if __name__ == "__main__":
    from glob import glob
    from pathlib import Path

    model = Predictor("output/tf_mobilenet/weights/best.onnx", imgsz=320)

    for file in glob("datasets/pole-cls/train/**", recursive=True):
        if Path(file).is_file():
            print(file)
            model(file)
