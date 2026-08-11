import torch
import torch.nn as nn

from .blocks import ConvBlock


class DetectionHead(nn.Module):
    """
    YOLO11-style decoupled detection head.

    Each feature level has:
        - Regression branch
        - Classification branch

    Regression:
        4 * reg_max channels

    Classification:
        num_classes channels
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        reg_max=16
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_max = reg_max

        # ====================================================
        # Regression branch
        # ====================================================

        self.reg_branch = nn.Sequential(

            ConvBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            ),

            ConvBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            ),

            nn.Conv2d(
                in_channels,
                4 * reg_max,
                kernel_size=1
            )
        )

        # ====================================================
        # Classification branch
        # ====================================================

        self.cls_branch = nn.Sequential(

            ConvBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            ),

            ConvBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            ),

            nn.Conv2d(
                in_channels,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Regression
        # ----------------------------------------------------

        box_output = self.reg_branch(x)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        cls_output = self.cls_branch(x)

        return {
            "box": box_output,
            "cls": cls_output
        }
    




class YOLO11Detect(nn.Module):
    """
    Multi-scale YOLO11-style detection head.

    Takes:
        F3
        F4
        F5

    Produces predictions for all three scales.
    """

    def __init__(
        self,
        num_classes,
        reg_max=16
    ):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max

        self.head_p3 = DetectionHead(
            in_channels=64,
            num_classes=num_classes,
            reg_max=reg_max
        )

        self.head_p4 = DetectionHead(
            in_channels=128,
            num_classes=num_classes,
            reg_max=reg_max
        )

        self.head_p5 = DetectionHead(
            in_channels=256,
            num_classes=num_classes,
            reg_max=reg_max
        )

    def forward(
        self,
        f3,
        f4,
        f5
    ):

        out_p3 = self.head_p3(
            f3
        )

        out_p4 = self.head_p4(
            f4
        )

        out_p5 = self.head_p5(
            f5
        )

        return {
            "p3": out_p3,
            "p4": out_p4,
            "p5": out_p5
        }