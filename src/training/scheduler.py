import torch


def build_scheduler(
    optimizer,
    config
):
    """
    Build learning-rate scheduler
    from configuration.
    """

    scheduler_name = (
        config["scheduler"]["name"]
        .lower()
    )

    epochs = config[
        "training"
    ]["epochs"]

    min_lr = config[
        "scheduler"
    ]["min_lr"]

    if scheduler_name == "cosine":

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr
            )
        )

    elif scheduler_name == "step":

        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        )

    elif scheduler_name == "none":

        scheduler = None

    else:

        raise ValueError(
            f"Unsupported scheduler: "
            f"{scheduler_name}"
        )

    return scheduler