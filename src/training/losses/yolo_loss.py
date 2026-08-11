import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_loss import iou_loss
from .dfl_loss import DistributionFocalLoss
from .assigner import (
    make_grid_points,
    decode_distribution,
    distances_to_boxes,
    assign_targets
)


class YOLOLoss(nn.Module):
    """
    YOLO-style detection loss.

    Components:

        Box loss
        Classification BCE loss
        Distribution Focal Loss
    """

    def __init__(
        self,
        num_classes,
        reg_max=16,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5
    ):
        super().__init__()

        self.num_classes = (
            num_classes
        )

        self.reg_max = (
            reg_max
        )

        self.box_weight = (
            box_weight
        )

        self.cls_weight = (
            cls_weight
        )

        self.dfl_weight = (
            dfl_weight
        )

        self.dfl_loss = (
            DistributionFocalLoss(
                reg_max=reg_max
            )
        )

    def forward(
        self,
        outputs,
        targets
    ):

        device = (
            outputs["p3"]["box"].device
        )

        total_box_loss = torch.zeros(
            (),
            device=device
        )

        total_cls_loss = torch.zeros(
            (),
            device=device
        )

        total_dfl_loss = torch.zeros(
            (),
            device=device
        )

        total_foreground = 0

        # ----------------------------------------------------
        # Detection scales
        # ----------------------------------------------------

        levels = {
            "p3": 8,
            "p4": 16,
            "p5": 32
        }

        for level, stride in (
            levels.items()
        ):

            box_pred = outputs[
                level
            ]["box"]

            cls_pred = outputs[
                level
            ]["cls"]

            batch_size, _, height, width = (
                box_pred.shape
            )

            # ------------------------------------------------
            # Grid points
            # ------------------------------------------------

            points = make_grid_points(
                height=height,
                width=width,
                stride=stride,
                device=device
            )

            # ------------------------------------------------
            # Decode box distributions
            # ------------------------------------------------

            distances = (
                decode_distribution(
                    box_pred,
                    self.reg_max
                )
            )

            pred_boxes = (
                distances_to_boxes(
                    points,
                    distances
                )
            )

            # ------------------------------------------------
            # Reshape classification
            # ------------------------------------------------

            cls_pred = cls_pred.permute(
                0,
                2,
                3,
                1
            ).reshape(
                batch_size,
                height * width,
                self.num_classes
            )

            # ------------------------------------------------
            # Process each image
            # ------------------------------------------------

            for batch_index in range(
                batch_size
            ):

                gt_boxes = targets[
                    batch_index
                ]["boxes"].to(
                    device
                )

                gt_labels = targets[
                    batch_index
                ]["labels"].to(
                    device
                )

                # ------------------------------------------------
                # Assign targets
                # ------------------------------------------------

                (
                    target_boxes,
                    target_labels,
                    foreground
                ) = assign_targets(
                    points=points,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    image_width=width * stride,
                    image_height=height * stride
                )

                # ------------------------------------------------
                # Classification target
                # ------------------------------------------------

                cls_target = torch.zeros_like(
                    cls_pred[
                        batch_index
                    ]
                )

                positive_indices = (
                    torch.where(
                        foreground
                    )[0]
                )

                if positive_indices.numel() > 0:

                    cls_target[
                        positive_indices,
                        target_labels[
                            positive_indices
                        ]
                    ] = 1.0

                # ------------------------------------------------
                # Classification loss
                # ------------------------------------------------

                cls_loss = (
                    F.binary_cross_entropy_with_logits(
                        cls_pred[
                            batch_index
                        ],
                        cls_target,
                        reduction="sum"
                    )
                )

                normalizer = max(
                    positive_indices.numel(),
                    1
                )

                cls_loss = (
                    cls_loss
                    / normalizer
                )

                total_cls_loss += (
                    cls_loss
                )

                # ------------------------------------------------
                # Box + DFL only for positives
                # ------------------------------------------------

                if positive_indices.numel() == 0:

                    continue

                total_foreground += (
                    positive_indices.numel()
                )

                positive_pred_boxes = (
                    pred_boxes[
                        batch_index,
                        positive_indices
                    ]
                )

                positive_target_boxes = (
                    target_boxes[
                        positive_indices
                    ]
                )

                # ------------------------------------------------
                # IoU loss
                # ------------------------------------------------

                box_loss = iou_loss(
                    positive_pred_boxes,
                    positive_target_boxes
                )

                total_box_loss += (
                    box_loss
                )

                # ------------------------------------------------
                # Target distances
                # ------------------------------------------------

                positive_points = (
                    points[
                        positive_indices
                    ]
                )

                target_distances = (
                    torch.stack(
                        [
                            positive_points[:, 0]
                            - positive_target_boxes[:, 0],

                            positive_points[:, 1]
                            - positive_target_boxes[:, 1],

                            positive_target_boxes[:, 2]
                            - positive_points[:, 0],

                            positive_target_boxes[:, 3]
                            - positive_points[:, 1]
                        ],
                        dim=-1
                    )
                )

                # ------------------------------------------------
                # Predicted distribution
                # ------------------------------------------------

                positive_box_logits = (
                    box_pred[
                        batch_index
                    ]
                    .permute(
                        1,
                        2,
                        0
                    )
                    .reshape(
                        height * width,
                        4,
                        self.reg_max
                    )
                )

                positive_box_logits = (
                    positive_box_logits[
                        positive_indices
                    ]
                )

                # DFL distances must be in stride units
                target_distances = (
                    target_distances
                    / stride
                )

                dfl_loss = (
                    self.dfl_loss(
                        positive_box_logits,
                        target_distances
                    )
                )

                total_dfl_loss += (
                    dfl_loss
                )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        num_scales = len(
            levels
        )

        total_box_loss = (
            total_box_loss
            / num_scales
        )

        total_cls_loss = (
            total_cls_loss
            / num_scales
        )

        total_dfl_loss = (
            total_dfl_loss
            / num_scales
        )

        # ----------------------------------------------------
        # Weighted total
        # ----------------------------------------------------

        total_loss = (
            self.box_weight
            * total_box_loss
            +
            self.cls_weight
            * total_cls_loss
            +
            self.dfl_weight
            * total_dfl_loss
        )

        return {
            "total": total_loss,

            "box": total_box_loss,

            "cls": total_cls_loss,

            "dfl": total_dfl_loss,

            "num_foreground":
                total_foreground
        }