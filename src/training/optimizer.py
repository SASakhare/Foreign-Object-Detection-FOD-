import torch


def build_optimizer(
    model,
    config
):
    """
    Build optimizer from configuration.
    """

    optimizer_name = (
        config["optimizer"]["name"]
        .lower()
    )

    lr = config[
        "optimizer"
    ]["learning_rate"]

    weight_decay = config[
        "optimizer"
    ]["weight_decay"]

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == "adamw":

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == "sgd":

        momentum = config[
            "optimizer"
        ].get(
            "momentum",
            0.937
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{optimizer_name}"
        )

    return optimizer