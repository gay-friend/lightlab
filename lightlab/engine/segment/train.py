import torch
import numpy as np
import math
import cv2
import contextlib

from lightlab.engine.trainer import BaseTrainer
from lightlab.utils import RANK
from lightlab.utils.loss import FocalLoss
from lightlab.utils.metrics import SegmentMetrics
from lightlab.data.dataset import SegmentDataset
from lightlab.utils.plotting import plot_results
from lightlab.models.segment import SegmentationModel
from lightlab.cfg import Settings


class SegmentTrainer(BaseTrainer):
    def __init__(
        self,
        cfg=Settings(),
        overrides=None,
        root="datasets/bushing_sleeve_od_abration",
        model_name="lightlab_mobilenetv3",
        scale="tiny",
        weights=None,
        names=None,
        nc=3,
    ):
        super().__init__(cfg, overrides, root, model_name, scale, weights, names, nc)
        self.task = "segment"

    def _setup_criterion(self):
        return FocalLoss()

    def build_dataset(self, root, mode="train", batch=None):
        return SegmentDataset(
            root=root,
            imgsz=self.args.imgsz,
            augment=mode == "train",
        )

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        batch["mask"] = batch["mask"].to(self.device, non_blocking=True).long()
        return batch

    def get_model(self):
        model = SegmentationModel(
            model_name=self.model_name,
            scale=self.scale,
            nc=self.nc,
            names=self.names,
            verbose=self.args.verbose and RANK == -1,
        )
        if self.weights:
            model.load(self.weights)
        return model

    def get_validator(self):
        self.loss_names = ["seg_loss"]

    def validate(self):
        metrics = SegmentMetrics(self.nc)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                imgs = batch["img"].to(self.device) / 255
                masks = batch["mask"].cpu().numpy()
                preds = self.model(imgs)
                preds = preds.detach().max(dim=1)[1].cpu().numpy()
                metrics.update(masks, preds)
        im: np.ndarray = imgs[0].cpu().numpy()
        im = (im * 255).astype(np.uint8).transpose(1, 2, 0)
        pred: np.ndarray = preds[0] * 255

        print(masks[0].max())
        mask: np.ndarray = masks[0] * 125
        mask = mask.astype(np.uint8)
        pred = pred.astype(np.uint8)

        from PIL import Image

        Image.fromarray(im).save(self.save_dir / "img.png")
        Image.fromarray(mask).save(self.save_dir / "mask.png")
        Image.fromarray(pred).save(self.save_dir / "pred.png")
        # use loss as fitness measure if not found
        metrics = metrics.get_results()
        fitness = metrics.pop("fitness", 0)
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        print(metrics)
        return metrics, fitness

    def progress_string(self):
        return ("\n" + "%11s" * (3 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Size",
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def update_pbar(self, pbar, batch, epoch, i):
        # Log
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
        losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
        if RANK in (-1, 0):
            pbar.set_description(
                ("%11s" * 2 + "%11.4g" * (1 + loss_len))
                % (
                    f"{epoch + 1}/{self.epochs}",
                    mem,
                    *losses,
                    batch["img"].shape[-1],
                )
            )
            if self.args.plots and i == 0:
                self.plot_training_samples(batch, i)

    def plot_training_samples(self, batch, ni):
        pass

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png


if __name__ == "__main__":
    trainer = SegmentTrainer(
        model_name="lightlab_mobilenetv3",
        root="datasets/bushing_sleeve_od_abration",
        overrides=dict(save_dir="output/segment", workers=0, imgsz=(512, 640)),
        nc=2,
    )
    trainer.train()
