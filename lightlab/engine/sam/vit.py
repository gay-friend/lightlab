import numpy as np

from lightlab.engine.onnx_predictor import OnnxPredictor


class Vit(OnnxPredictor):
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    def __init__(self, model_path, device="cuda", warmup_epoch=1, **kwargs) -> None:
        super().__init__(model_path, 1024, 1, device, 0, **kwargs)

        self.imgsz = self.input_shape[2:]
        self.batch = self.input_shape[0]
        if warmup_epoch:
            self.warmup(warmup_epoch)

    def transform(self, img: np.ndarray) -> np.ndarray:
        img = super().transform(img)
        img = (img - self.mean) / self.std
        return img

    def __call__(self, imgs: np.ndarray):
        preds = super().__call__(imgs)
        return np.array(preds)
