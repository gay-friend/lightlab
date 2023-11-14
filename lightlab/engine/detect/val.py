import numpy as np
import torch

from lightlab.engine.validator import BaseValidator
from lightlab.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from lightlab.utils.plotting import plot_images
from lightlab.utils.ops import non_max_suppression, scale_boxes, xywh2xyxy, xyxy2xywh
from lightlab.utils import LOGGER
from lightlab.utils.torch_utils import de_parallel
from lightlab.utils.plotting import output_to_target


class DetectionValidator(BaseValidator):
    def __init__(self, dataloader=None, pbar=None, args=None):
        super().__init__(dataloader, pbar, args)
        self.nt_per_class = None
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (
            batch["img"].half() if self.args.half else batch["img"].float()
        ) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch

    def init_metrics(self, model):
        self.class_map = list(range(1000))
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        return non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch["ori_shape"][si]
            correct_bboxes = torch.zeros(
                npr, self.niou, dtype=torch.bool, device=self.device
            )  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append(
                        (
                            correct_bboxes,
                            *torch.zeros((2, 0), device=self.device),
                            cls.squeeze(-1),
                        )
                    )
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, labels=cls.squeeze(-1)
                        )
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(
                batch["img"][si].shape[1:],
                predn[:, :4],
                shape,
                ratio_pad=batch["ratio_pad"][si],
            )  # native-space pred

            # Evaluate
            if nl:
                height, width = batch["img"].shape[2:]
                tbox = xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device
                )  # target boxes
                scale_boxes(
                    batch["img"][si].shape[1:],
                    tbox,
                    shape,
                    ratio_pad=batch["ratio_pad"][si],
                )  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append(
                (correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1))
            )  # (conf, pcls, tcls)

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(
            stats[-1].astype(int), minlength=self.nc
        )  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        print(
            pf
            % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        )
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                "No labels found in set, can not compute metrics without labels"
            )

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.seen,
                        self.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

        if self.args.plots:
            self.confusion_matrix.plot(
                save_dir=self.save_dir,
                names=self.names.values(),
                normalize=True,
            )

    def _process_batch(self, detections, labels):
        iou = box_iou(labels[:, 1:], detections[:, :4])
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def plot_val_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
        )

    def plot_predictions(self, batch, preds, ni):
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
        )  # pred
