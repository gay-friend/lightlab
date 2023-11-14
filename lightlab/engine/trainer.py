import math
import os
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from lightlab.cfg import Settings, get_cfg
from lightlab.utils.torch_utils import init_seeds, one_cycle, EarlyStopping, ModelEMA
from lightlab.utils import RANK, LOGGER, TQDM
from lightlab.utils.torch_utils import torch_distributed_zero_first
from lightlab.data.build import build_dataloader


class BaseTrainer:
    def __init__(
        self,
        cfg: Settings = Settings(),
        overrides=None,
        root="datasets/pole-cls",
        model_name="yolov8",
        scale="tiny",
        weights=None,
        names=None,
        nc=3,
    ):
        self.args = get_cfg(cfg, overrides)
        self.device = self.args.device
        self.task = "unknow"
        self.model_name = model_name
        self.scale = scale
        self.root = Path(root)
        self.weights = weights
        self.names = {i: str(i) for i in range(nc)} if names is None else names
        self.nc = nc

        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK)

        # Dirs
        self.save_dir = Path(self.args.save_dir)
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = (
            self.wdir / "last.pt",
            self.wdir / "best.pt",
        )  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0

        # Device
        if self.device in ("cpu", "mps"):
            # faster CPU training as time dominated by inference, not dataloading
            self.args.workers = 0

        # Model and Dataset
        self.model = None
        self.trainset, self.testset = self.root / "train", self.root / "val"
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):
            world_size = len(self.args.device.split(","))
        elif isinstance(
            self.args.device, (tuple, list)
        ):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif (
            torch.cuda.is_available()
        ):  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            # if self.args.rect:
            #     LOGGER.warning(
            #         "WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'"
            #     )
            #     self.args.rect = False
            # if self.args.batch == -1:
            #     LOGGER.warning(
            #         "WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
            #         "default 'batch=16'"
            #     )
            #     self.args.batch = 16

            # # Command
            # cmd, file = generate_ddp_command(world_size, self)
            # try:
            #     LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
            #     subprocess.run(cmd, check=True)
            # except Exception as e:
            #     raise e
            # finally:
            #     ddp_cleanup(self, str(file))
            pass

        else:
            self._do_train(world_size)

    def _setup_ddp(self, world_size):
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.model = self.get_model()
        self.model.names = self.names
        self.model = self.model.to(self.device)

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False

        if RANK > -1 and world_size > 1:  # DDP
            # broadcast the tensor from rank 0 to all other ranks (returns None)

            dist.broadcast(self.amp, src=0)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])

        # Batch size
        # if (
        #     self.batch_size == -1 and RANK == -1
        # ):  # single-GPU only, estimate best batch size
        #     self.args.batch = self.batch_size = check_train_batch_size(
        #         self.model, self.args.imgsz, self.amp
        #     )

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=batch_size, rank=RANK, mode="train"
        )
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size * 2, rank=RANK, mode="val"
            )
            self.validator = self.get_validator()
            # metric_keys = self.validator.metrics.keys + self.label_loss_items(
            #     prefix="val"
            # )
            # self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        # accumulate loss before optimizing
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = (
            self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        )  # scale weight_decay
        iterations = (
            math.ceil(
                len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)
            )
            * self.epochs
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = (
                lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf
            )  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.criterion = self._setup_criterion()

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        # warmup iterations
        nw = (
            max(round(self.args.warmup_epochs * nb), 100)
            if self.args.warmup_epochs > 0
            else -1
        )
        last_opt_step = -1
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {self.save_dir}\n"
            f"Starting training for {self.epochs} epochs..."
        )
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)

            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info("Closing dataloader mosaic")
                if hasattr(self.train_loader.dataset, "mosaic"):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, "close_mosaic"):
                    self.train_loader.dataset.close_mosaic()
                self.train_loader.reset()

            if RANK in (-1, 0):
                print(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            # self.optimizer.zero_grad()
            for i, batch in pbar:
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(
                        1,
                        np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round(),
                    )
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if j == 0 else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [self.args.warmup_momentum, self.args.momentum]
                            )
                else:
                    self.accumulate = 1

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch["img"])
                    self.loss, self.loss_items = self.criterion(preds, batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1)
                        if self.tloss is not None
                        else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                self.update_pbar(pbar, batch, epoch, i)

            self.lr = {
                f"lr/pg{ir}": x["lr"]
                for ir, x in enumerate(self.optimizer.param_groups)
            }  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            if RANK in (-1, 0):
                # Validation
                self.ema.update_attr(
                    self.model,
                    include=["nc", "args", "names", "stride", "class_weights"],
                )

                self.metrics, self.fitness = self.validate()
                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss),
                        **self.metrics,
                        **self.lr,
                    }
                )
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                self.save_model()

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(
                    broadcast_list, 0
                )  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            if self.args.plots:
                self.plot_metrics()
        torch.cuda.empty_cache()

    def update_pbar(self, pbar: TQDM, batch, epoch, i):
        # Log
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
        losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
        if RANK in (-1, 0):
            pbar.set_description(
                ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                % (
                    f"{epoch + 1}/{self.epochs}",
                    mem,
                    *losses,
                    batch["cls"].shape[0],
                    batch["img"].shape[-1],
                )
            )
            if self.args.plots and i == 0:
                self.plot_training_samples(batch, i)

    def save_model(self):
        # Save last and best
        # ckpt = self.model.state_dict()
        ckpt = self.model
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (
            (self.save_period > 0)
            and (self.epoch > 0)
            and (self.epoch % self.save_period == 0)
        ):
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        return batch

    def validate(self):
        metrics = self.validator(self)
        # use loss as fitness measure if not found
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # init dataset *.cache only once if DDP
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        workers = self.args.workers if mode == "train" else self.args.workers * 2
        loader = build_dataloader(dataset, batch_size, workers, rank=rank)
        return loader

    def _setup_criterion(self):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_model(self):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        raise NotImplementedError("get_validator function not implemented in trainer")

    def build_dataset(self, root, mode="train", batch=None):
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def build_targets(self, preds, targets):
        pass

    def progress_string(self):
        return ""

    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = (
            ""
            if self.csv.exists()
            else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")
        )  # header
        with open(self.csv, "a") as f:
            f.write(
                s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n"
            )

    def plot_metrics(self):
        pass

    def build_optimizer(
        self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5
    ):
        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                "optimizer: 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                "determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(
                0.002 * 5 / (4 + nc), 6
            )  # lr0 fit equation to 6 decimal places
            name, lr, momentum = (
                ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            )
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
            )

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"optimizer: {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
