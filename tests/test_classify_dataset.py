from lightlab.data.dataset import ClassificationDataset
import cv2
import numpy as np
import torch

if __name__ == "__main__":
    dataset = ClassificationDataset(root="datasets/pole-cls/train", augment=True)
    for i in range(5):
        batch = dataset[i]
        img = batch["img"]
        img: torch.Tensor = img * 255
        img = img.permute(1, 2, 0)
        img = img.numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape, img.max())
        cv2.imwrite(f"output/batch_{i}.png", img)
