from pathlib import Path
import json
import torch


class TrainingHistory:

    def __init__(self):

        self.history = []

    def update(
        self,
        epoch,
        metrics
    ):

        record = {
            "epoch": epoch,
            **metrics
        }

        self.history.append(
            record
        )

    def save(
        self,
        path
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.history,
                f,
                indent=4
            )




class CheckpointManager:

    def __init__(
        self,
        save_dir
    ):

        self.save_dir = Path(
            save_dir
        )

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        metrics,
        filename
    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict()
                if optimizer is not None
                else None,

            "scheduler_state_dict":
                scheduler.state_dict()
                if scheduler is not None
                else None,

            "metrics": metrics
        }

        path = (
            self.save_dir /
            filename
        )

        torch.save(
            checkpoint,
            path
        )

        return path