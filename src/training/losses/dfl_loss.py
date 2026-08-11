import torch
import torch.nn.functional as F


class DistributionFocalLoss(
    torch.nn.Module
):
    """
    Distribution Focal Loss.

    Used for discretized bounding-box regression.

    For each side:
        left
        top
        right
        bottom

    the model predicts a probability distribution
    over [0, reg_max - 1].
    """

    def __init__(
        self,
        reg_max=16
    ):
        super().__init__()

        self.reg_max = reg_max

    def forward(
        self,
        pred,
        target
    ):
        """
        Args:
            pred:
                [N, 4, reg_max]

            target:
                [N, 4]

        Returns:
            scalar DFL loss
        """

        if pred.numel() == 0:

            return pred.sum() * 0.0

        target = target.clamp(
            min=0,
            max=self.reg_max - 1 - 1e-4
        )

        target_left = (
            target.floor()
            .long()
        )

        target_right = (
            target_left + 1
        ).clamp(
            max=self.reg_max - 1
        )

        weight_right = (
            target
            - target_left.float()
        )

        weight_left = (
            1.0
            - weight_right
        )

        pred = pred.reshape(
            -1,
            self.reg_max
        )

        target_left = (
            target_left.reshape(-1)
        )

        target_right = (
            target_right.reshape(-1)
        )

        weight_left = (
            weight_left.reshape(-1)
        )

        weight_right = (
            weight_right.reshape(-1)
        )

        loss_left = F.cross_entropy(
            pred,
            target_left,
            reduction="none"
        )

        loss_right = F.cross_entropy(
            pred,
            target_right,
            reduction="none"
        )

        loss = (
            loss_left * weight_left
            +
            loss_right * weight_right
        )

        return loss.mean()