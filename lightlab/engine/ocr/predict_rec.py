import numpy as np

from lightlab.engine.onnx_predictor import OnnxPredictor


def get_character_dict():
    with open("assets/ppocr/ppocr_keys_v1.txt", encoding="utf-8") as f:
        return f.read().splitlines()


character_dict = get_character_dict()


class PredictRec(OnnxPredictor):
    def __init__(
        self,
        model_path,
        imgsz=[48, 320],
        batch=6,
        device="cpu",
        warmup_epoch=1,
    ) -> None:
        super().__init__(model_path, imgsz, batch, device, warmup_epoch)
        self.character_type = "ch"
        self.is_remove_duplicate = True
        self.character = ["blank"] + character_dict + [" "]

    def transform(self, img: np.ndarray) -> np.ndarray:
        img = super().transform(img).astype(np.float32)
        import cv2

        cv2.imwrite("ppocr_re.png", img)
        img = img / 255
        img -= 0.5
        img /= 0.5
        return img.astype(np.float32)

    def preprocess(self, img: np.ndarray):
        if isinstance(img, list) and self.character_type == "ch":
            max_ratio = 1
            for im in img:
                h, w = im.shape[:2]
                ratio = w / h
                if ratio > max_ratio:
                    max_ratio = ratio
            self.imgsz[1] = int(32 * max_ratio)
        return super().preprocess(img)

    def postprocess(self, preds: np.ndarray):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        ignored_tokens = [0]

        batch_size = len(preds_idx)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            pred = preds_idx[batch_idx]
            for idx in range(len(pred)):
                if pred[idx] in ignored_tokens:
                    continue
                if self.is_remove_duplicate:
                    if idx > 0 and pred[idx - 1] == pred[idx]:
                        continue
                char_list.append(self.character[int(pred[idx])])
                conf_list.append(preds_prob[batch_idx][idx])
            text = "".join(char_list)
            conf = np.mean(conf_list) if conf_list else np.nan
            result_list.append((text, conf))

        return result_list


if __name__ == "__main__":
    predictor = PredictRec("assets/ppocr/ppocrv4_rec.onnx", [48, 320])
    out = predictor("assets/images/ppocr_cls.png")
    print(out)
