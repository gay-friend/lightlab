import numpy as np

from lightlab.engine.onnx_predictor import OnnxPredictor


class PredictCls(OnnxPredictor):
    def __init__(
        self,
        model_path,
        imgsz=(48, 192),
        batch=6,
        device="cpu",
        warmup_epoch=1,
        label_list=("0", "180"),
        cls_thresh=0.9,
    ) -> None:
        super().__init__(model_path, imgsz, batch, device, warmup_epoch)
        self.label_list = label_list
        self.cls_thresh = cls_thresh

    def transform(self, img: np.ndarray) -> np.ndarray:
        img = super().transform(img).astype(np.float32) / 255
        img -= 0.5
        img /= 0.5
        return img

    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        pred_idxs = outputs.argmax(axis=1)
        return [
            (self.label_list[idx], float(outputs[i, idx]))
            for i, idx in enumerate(pred_idxs)
        ]


if __name__ == "__main__":
    predictor = PredictCls("assets/ppocr/ppocrv2_cls.onnx", (48, 192))
    out = predictor("assets/images/ppocr_cls.png")
    print(out)
