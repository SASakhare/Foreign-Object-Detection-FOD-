import torch
from pathlib import Path

from .utils import (
    TrainingHistory,
    CheckpointManager
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        config
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config

        # ====================================================
        # DEVICE
        # ====================================================

        device_name = (
            config["training"]
            .get(
                "device",
                "cuda"
            )
        )

        if (
            device_name == "cuda"
            and torch.cuda.is_available()
        ):

            self.device = torch.device(
                "cuda"
            )

        else:

            self.device = torch.device(
                "cpu"
            )

        self.model = self.model.to(
            self.device
        )

        # ====================================================
        # AMP
        # ====================================================

        self.use_amp = (
            config["training"]
            .get(
                "amp",
                True
            )
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp
        )

        # ====================================================
        # GRADIENT CLIPPING
        # ====================================================

        self.gradient_clip = (
            config["training"]
            .get(
                "gradient_clip",
                None
            )
        )

        # ====================================================
        # TRAINING HISTORY
        # ====================================================

        self.history = TrainingHistory()

        # ====================================================
        # CHECKPOINT MANAGER
        # ====================================================

        checkpoint_dir = (
            config["checkpoint"]
            .get(
                "save_dir",
                "checkpoints"
            )
        )

        self.checkpoint_manager = (
            CheckpointManager(
                checkpoint_dir
            )
        )

        # ====================================================
        # BEST VALIDATION LOSS
        # ====================================================

        # Validation loss is minimized.
        self.best_metric = float("inf")

    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train_one_epoch(
        self,
        epoch
    ):

        self.model.train()

        # ----------------------------------------------------
        # Running losses
        # ----------------------------------------------------

        running_loss = 0.0

        running_box_loss = 0.0

        running_cls_loss = 0.0

        running_dfl_loss = 0.0

        num_batches = len(
            self.train_loader
        )

        # ----------------------------------------------------
        # Batch loop
        # ----------------------------------------------------

        for batch_idx, batch in enumerate(
            self.train_loader
        ):

            images, targets = batch

            images = images.to(
                self.device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward + Loss
            # ------------------------------------------------

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):

                outputs = self.model(
                    images
                )

                # YOLOLoss returns dictionary
                #
                # {
                #     "total": ...,
                #     "box": ...,
                #     "cls": ...,
                #     "dfl": ...,
                #     "num_foreground": ...
                # }

                loss_dict = self.criterion(
                    outputs,
                    targets
                )

                # Actual loss used for backward
                loss = loss_dict["total"]

            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            if self.use_amp:

                self.scaler.scale(
                    loss
                ).backward()

                # --------------------------------------------
                # Gradient clipping
                # --------------------------------------------

                if (
                    self.gradient_clip
                    is not None
                ):

                    self.scaler.unscale_(
                        self.optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # --------------------------------------------
                # Optimizer step
                # --------------------------------------------

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                loss.backward()

                # --------------------------------------------
                # Gradient clipping
                # --------------------------------------------

                if (
                    self.gradient_clip
                    is not None
                ):

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # --------------------------------------------
                # Optimizer step
                # --------------------------------------------

                self.optimizer.step()

            # =================================================
            # ACCUMULATE LOSSES
            # =================================================

            running_loss += (
                loss.item()
            )

            running_box_loss += (
                loss_dict["box"].item()
            )

            running_cls_loss += (
                loss_dict["cls"].item()
            )

            running_dfl_loss += (
                loss_dict["dfl"].item()
            )

            # =================================================
            # BATCH LOGGING
            # =================================================

            print_frequency = (
                self.config["logging"]
                .get(
                    "print_frequency",
                    20
                )
            )

            if (
                batch_idx % print_frequency
                == 0
            ):

                print(
                    f"Epoch {epoch} | "
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{num_batches} | "
                    f"Total: "
                    f"{loss.item():.4f} | "
                    f"Box: "
                    f"{loss_dict['box'].item():.4f} | "
                    f"Cls: "
                    f"{loss_dict['cls'].item():.4f} | "
                    f"DFL: "
                    f"{loss_dict['dfl'].item():.4f} | "
                    f"FG: "
                    f"{loss_dict['num_foreground']}"
                )

        # ====================================================
        # EPOCH AVERAGES
        # ====================================================

        denominator = max(
            num_batches,
            1
        )

        epoch_loss = (
            running_loss /
            denominator
        )

        epoch_box_loss = (
            running_box_loss /
            denominator
        )

        epoch_cls_loss = (
            running_cls_loss /
            denominator
        )

        epoch_dfl_loss = (
            running_dfl_loss /
            denominator
        )

        return {

            "loss":
                epoch_loss,

            "box":
                epoch_box_loss,

            "cls":
                epoch_cls_loss,

            "dfl":
                epoch_dfl_loss
        }

    # ========================================================
    # VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        # ----------------------------------------------------
        # Running losses
        # ----------------------------------------------------

        running_loss = 0.0

        running_box_loss = 0.0

        running_cls_loss = 0.0

        running_dfl_loss = 0.0

        num_batches = len(
            self.val_loader
        )

        # ----------------------------------------------------
        # Validation loop
        # ----------------------------------------------------

        for images, targets in (
            self.val_loader
        ):

            images = images.to(
                self.device,
                non_blocking=True
            )

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):

                outputs = self.model(
                    images
                )

                loss_dict = self.criterion(
                    outputs,
                    targets
                )

                loss = loss_dict["total"]

            # ------------------------------------------------
            # Accumulate
            # ------------------------------------------------

            running_loss += (
                loss.item()
            )

            running_box_loss += (
                loss_dict["box"].item()
            )

            running_cls_loss += (
                loss_dict["cls"].item()
            )

            running_dfl_loss += (
                loss_dict["dfl"].item()
            )

        # ====================================================
        # AVERAGE
        # ====================================================

        denominator = max(
            num_batches,
            1
        )

        val_loss = (
            running_loss /
            denominator
        )

        val_box_loss = (
            running_box_loss /
            denominator
        )

        val_cls_loss = (
            running_cls_loss /
            denominator
        )

        val_dfl_loss = (
            running_dfl_loss /
            denominator
        )

        return {

            "loss":
                val_loss,

            "box":
                val_box_loss,

            "cls":
                val_cls_loss,

            "dfl":
                val_dfl_loss
        }

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    def fit(self):

        epochs = (
            self.config["training"]
            ["epochs"]
        )

        # ====================================================
        # EPOCH LOOP
        # ====================================================

        for epoch in range(
            1,
            epochs + 1
        ):

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Epoch {epoch}/{epochs}"
            )

            print(
                "=" * 70
            )

            # =================================================
            # TRAIN
            # =================================================

            train_metrics = (
                self.train_one_epoch(
                    epoch
                )
            )

            # =================================================
            # VALIDATION
            # =================================================

            val_metrics = (
                self.validate()
            )

            # =================================================
            # SCHEDULER
            # =================================================

            if self.scheduler is not None:

                self.scheduler.step()

            # =================================================
            # CURRENT LEARNING RATE
            # =================================================

            current_lr = (
                self.optimizer
                .param_groups[0]
                ["lr"]
            )

            # =================================================
            # COMBINE METRICS
            # =================================================

            metrics = {

                # ----------------------------
                # Training
                # ----------------------------

                "train_loss":
                    train_metrics["loss"],

                "train_box_loss":
                    train_metrics["box"],

                "train_cls_loss":
                    train_metrics["cls"],

                "train_dfl_loss":
                    train_metrics["dfl"],

                # ----------------------------
                # Validation
                # ----------------------------

                "val_loss":
                    val_metrics["loss"],

                "val_box_loss":
                    val_metrics["box"],

                "val_cls_loss":
                    val_metrics["cls"],

                "val_dfl_loss":
                    val_metrics["dfl"],

                # ----------------------------
                # Learning rate
                # ----------------------------

                "learning_rate":
                    current_lr
            }

            # =================================================
            # SAVE HISTORY
            # =================================================

            self.history.update(
                epoch,
                metrics
            )

            # =================================================
            # PRINT EPOCH SUMMARY
            # =================================================

            print(
                "\n"
                "--------------- "
                "Training "
                "---------------"
            )

            print(
                f"Train Total : "
                f"{train_metrics['loss']:.4f}"
            )

            print(
                f"Train Box   : "
                f"{train_metrics['box']:.4f}"
            )

            print(
                f"Train Cls   : "
                f"{train_metrics['cls']:.4f}"
            )

            print(
                f"Train DFL   : "
                f"{train_metrics['dfl']:.4f}"
            )

            print(
                "\n"
                "--------------- "
                "Validation "
                "---------------"
            )

            print(
                f"Val Total   : "
                f"{val_metrics['loss']:.4f}"
            )

            print(
                f"Val Box     : "
                f"{val_metrics['box']:.4f}"
            )

            print(
                f"Val Cls     : "
                f"{val_metrics['cls']:.4f}"
            )

            print(
                f"Val DFL     : "
                f"{val_metrics['dfl']:.4f}"
            )

            print(
                f"\nLearning Rate: "
                f"{current_lr:.8f}"
            )

            # =================================================
            # SAVE LAST CHECKPOINT
            # =================================================

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics,
                filename="last.pt"
            )

            # =================================================
            # SAVE BEST CHECKPOINT
            # =================================================

            if (
                val_metrics["loss"]
                < self.best_metric
            ):

                self.best_metric = (
                    val_metrics["loss"]
                )

                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    filename="best.pt"
                )

                print(
                    "\nBest model saved."
                )

        # ====================================================
        # SAVE TRAINING HISTORY
        # ====================================================

        self.history.save(
            Path(
                self.checkpoint_manager
                .save_dir
            ) / "history.json"
        )

        print(
            "\nTraining completed."
        )