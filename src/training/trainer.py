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

        self.train_loader = (
            train_loader
        )

        self.val_loader = (
            val_loader
        )

        self.optimizer = (
            optimizer
        )

        self.scheduler = (
            scheduler
        )

        self.criterion = (
            criterion
        )

        self.config = config

        # ------------------------------------------------
        # Device
        # ------------------------------------------------

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

        # ------------------------------------------------
        # AMP
        # ------------------------------------------------

        self.use_amp = (
            config["training"]
            .get(
                "amp",
                True
            )
            and self.device.type == "cuda"
        )

        self.scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=self.use_amp
            )
        )

        # ------------------------------------------------
        # Gradient clipping
        # ------------------------------------------------

        self.gradient_clip = (
            config["training"]
            .get(
                "gradient_clip",
                None
            )
        )

        # ------------------------------------------------
        # History
        # ------------------------------------------------

        self.history = (
            TrainingHistory()
        )

        # ------------------------------------------------
        # Checkpoint manager
        # ------------------------------------------------

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

        self.best_metric = float(
            "-inf"
        )

    # ====================================================
    # TRAIN ONE EPOCH
    # ====================================================

    def train_one_epoch(
        self,
        epoch
    ):

        self.model.train()

        running_loss = 0.0

        num_batches = len(
            self.train_loader
        )

        for batch_idx, batch in enumerate(
            self.train_loader
        ):

            images, targets = batch

            images = images.to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward + loss
            # ------------------------------------------------

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):

                outputs = self.model(
                    images
                )

                loss = self.criterion(
                    outputs,
                    targets
                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            if self.use_amp:

                self.scaler.scale(
                    loss
                ).backward()

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

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                loss.backward()

                if (
                    self.gradient_clip
                    is not None
                ):

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.optimizer.step()

            running_loss += (
                loss.item()
            )

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

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
                    f"Epoch "
                    f"{epoch} | "
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{num_batches} | "
                    f"Loss: "
                    f"{loss.item():.4f}"
                )

        epoch_loss = (
            running_loss /
            max(num_batches, 1)
        )

        return epoch_loss

    # ====================================================
    # VALIDATION
    # ====================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        num_batches = len(
            self.val_loader
        )

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

                loss = self.criterion(
                    outputs,
                    targets
                )

            running_loss += (
                loss.item()
            )

        val_loss = (
            running_loss /
            max(num_batches, 1)
        )

        return val_loss

    # ====================================================
    # FIT
    # ====================================================

    def fit(self):

        epochs = (
            self.config["training"]
            ["epochs"]
        )

        for epoch in range(
            1,
            epochs + 1
        ):

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Epoch "
                f"{epoch}/{epochs}"
            )

            print(
                "=" * 60
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            train_loss = (
                self.train_one_epoch(
                    epoch
                )
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_loss = (
                self.validate()
            )

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            if self.scheduler is not None:

                self.scheduler.step()

            # ------------------------------------------------
            # Current LR
            # ------------------------------------------------

            current_lr = (
                self.optimizer
                .param_groups[0]
                ["lr"]
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = {

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "learning_rate":
                    current_lr
            }

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            self.history.update(
                epoch,
                metrics
            )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"\nTrain Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Val Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"LR: "
                f"{current_lr:.8f}"
            )

            # ------------------------------------------------
            # Save last
            # ------------------------------------------------

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics,
                filename="last.pt"
            )

            # ------------------------------------------------
            # Save best
            # ------------------------------------------------

            if val_loss < self.best_metric:

                self.best_metric = val_loss

                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    filename="best.pt"
                )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        self.history.save(
            Path(
                self.checkpoint_manager
                .save_dir
            ) / "history.json"
        )