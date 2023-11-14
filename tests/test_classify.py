from lightlab.models.classify import ClassificationModel
import torch

if __name__ == "__main__":
    im = torch.zeros((3, 3, 640, 640))

    # for model_name in ("yolov8_cls", "tf_mobilenetv3_large_075", "tf_efficientnet_b0"):
    for model_name in ["tf_mobilenetv3_large_075"]:
        model = ClassificationModel(model_name)
        pred = model(im)
        print(model)
