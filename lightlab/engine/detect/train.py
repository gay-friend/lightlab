from lightlab.engine.trainer import BaseTrainer
from lightlab.engine.detect.val import DetectionValidator
from lightlab.utils.torch_utils import de_parallel
from lightlab.utils import RANK
from lightlab.utils.loss import DetectionLoss
from lightlab.data.dataset import YoloDataset
from lightlab.utils.plotting import plot_images, plot_results
from lightlab.models.detect import DetectionModel
from lightlab.cfg import Settings


class DetectionTrainer(BaseTrainer):
    def __init__(
        self,
        cfg=Settings(),
        overrides=None,
        root="datasets/coco128",
        model_name="yolov8",
        scale="tiny",
        weights=None,
        names=None,
        nc=3,
    ):
        super().__init__(cfg, overrides, root, model_name, scale, weights, names, nc)
        self.task = "detect"

    def _setup_criterion(self):
        return DetectionLoss(self.model, self.args)

    def build_dataset(self, root, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return YoloDataset(
            root=root,
            cfg=self.args,
            augment=mode == "train",
            stride=gs,
            rect=mode == "val",
        )

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def get_model(self):
        model = DetectionModel(
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
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(self.test_loader, args=self.args)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [
                round(float(x), 5) for x in loss_items
            ]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
        )

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png


if __name__ == "__main__":
    # trainer = ClassificationTrainer(model_name="tf_efficientnet_b0")
    trainer = DetectionTrainer(
        model_name="yolov8",
        overrides=dict(
            save_dir="output/coco128",
            workers=0,
        ),
        nc=80,
        weights="assets/yolov8n-dict.pt",
    )
    trainer.train()
