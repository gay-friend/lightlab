import onnxruntime as ort
import numpy as np
import cv2
import math
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from lightlab.utils import LOGGER


class OnnxPredictor:
    pad_val = 0

    def __init__(
        self,
        model_path,
        imgsz=640,
        batch=2,
        device="cpu",
        warmup_epoch=1,
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.warmup_epoch = warmup_epoch
        self.imgsz = imgsz
        self.batch = batch

        if isinstance(imgsz, int):
            self.imgsz = [imgsz, imgsz]

        opt = ort.SessionOptions()
        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")
        LOGGER.info(f"loading model from {model_path}...")
        self.session = ort.InferenceSession(
            model_path, opt, providers=provider, **kwargs
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        if warmup_epoch:
            self.warmup(warmup_epoch)

    def warmup(self, epoch) -> None:
        x = np.random.random((self.batch, 3, *self.imgsz)).astype(np.float32)
        LOGGER.info("start warmup!")
        for i in tqdm(range(epoch)):
            self.session.run(None, {self.input_name: x})
        LOGGER.info("warmup finish!")

    def transform(self, img: np.ndarray, pad=True, stride=None) -> np.ndarray:
        h, w = img.shape[:2]
        h0, w0 = self.imgsz

        # 补最小的尺寸，再pad
        # h / w = h0 / w0
        ratio = w / h
        if h > w:
            new_h = h0
            new_w = int(new_h * ratio)
        else:
            new_w = w0
            new_h = int(new_w / ratio)

        if new_w > w0:
            ratio = w0 / new_w
            new_w = w0
            new_h = int(new_w * ratio)
        elif new_h > h0:
            ratio = h0 / new_h
            new_h = h0
            new_w = int(new_h * ratio)

        if stride is not None:
            new_h = max(int(round(new_h / 32) * 32), 32)
            new_h = max(int(round(new_h / 32) * 32), 32)

        img = cv2.resize(img, (new_w, new_h))
        if pad:
            img = np.pad(
                img,
                ((0, h0 - new_h), (0, w0 - new_w), (0, 0)),
                "constant",
                constant_values=(self.pad_val, self.pad_val),
            )
        return img

    def postprocess(self, outputs: np.ndarray):
        return outputs

    def preprocess(self, imgs: np.ndarray):
        self.ori_shapes = []
        for i, im in enumerate(imgs):
            self.ori_shapes.append(im.shape)
            im = self.transform(im)
            imgs[i] = im

        imgs = np.array(imgs)
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
        return imgs

    def __call__(self, imgs: np.ndarray):
        outputs = []
        if isinstance(imgs, (Path, str)):
            img = Image.open(imgs).convert("RGB")
            img = np.array(img, dtype=np.uint8)
            imgs = [img]
        elif isinstance(imgs, np.ndarray):
            imgs = [imgs]

        for i in range(math.ceil(len(imgs) / self.batch)):
            start = i * self.batch
            end = (i + 1) * self.batch
            batch_imgs = self.preprocess(imgs[start:end])
            output = self.session.run(None, {self.input_name: batch_imgs})[0]
            outputs.extend(self.postprocess(output))

        return outputs


if __name__ == "__main__":
    model = OnnxPredictor("output/coco128/weights/best.onnx")
    file = "assets/images/bus.jpg"
    im = cv2.imread(file)
    out = model(im)
    print(out)
