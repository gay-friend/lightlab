import torch

from lightlab.models.classify import ClassificationModel
from lightlab.engine.trainer import BaseTrainer
from lightlab.utils.plotting import plot_images, plot_results
from lightlab.data.dataset import ClassificationDataset
from lightlab.engine.classify.val import ClassificationValidator
from lightlab.utils.loss import ClassificationLoss
from lightlab.cfg import Settings


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        cfg=Settings(),
        overrides=None,
        root="datasets/pole-cls",
        model_name="yolov8_cls",
        scale="tiny",
        weights=None,
        names=None,
        nc=3,
    ):
        super().__init__(cfg, overrides, root, model_name, scale, weights, names, nc)
        self.task = "classify"

    def get_model(self):
        model = ClassificationModel(
            self.model_name, self.scale, nc=self.nc, names=self.names
        )
        if self.weights:
            model.load(self.weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def build_dataset(self, root, mode="train", batch=None):
        return ClassificationDataset(root=root, args=self.args, augment=mode == "train")

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def _setup_criterion(self):
        return ClassificationLoss()

    def get_validator(self):
        self.loss_names = ["loss"]
        return ClassificationValidator(self.test_loader, args=self.args)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        plot_results(file=self.csv, classify=True)  # save results.png

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(
                -1
            ),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"train_batch{ni}.jpg",
            # norm=True,
        )


if __name__ == "__main__":
    # trainer = ClassificationTrainer(model_name="tf_efficientnet_b0")
    trainer = ClassificationTrainer(
        model_name="tf_mobilenetv3_large_075", overrides=dict(imgsz=320, epoch=50)
    )
    trainer.train()
