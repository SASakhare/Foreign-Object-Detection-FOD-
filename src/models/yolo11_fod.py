import torch
import torch.nn as nn

from .backbone import YOLO11Backbone
from .neck import YOLO11Neck
from .head import YOLO11Detect


class YOLO11FOD(nn.Module):
    """
    YOLO11n-style object detector for FOD detection.

    Architecture:

        Input
          ↓
        Backbone
          ↓
        P3, P4, P5
          ↓
        FPN + PAN Neck
          ↓
        F3, F4, F5
          ↓
        Detection Head
          ↓
        Box + Classification predictions
    """

    def __init__(
        self,
        num_classes,
        reg_max=16
    ):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max

        # ====================================================
        # BACKBONE
        # ====================================================

        self.backbone = YOLO11Backbone()

        # ====================================================
        # NECK
        # ====================================================

        self.neck = YOLO11Neck()

        # ====================================================
        # DETECTION HEAD
        # ====================================================

        self.head = YOLO11Detect(
            num_classes=num_classes,
            reg_max=reg_max
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        p3, p4, p5 = self.backbone(
            x
        )

        # ----------------------------------------------------
        # Neck
        # ----------------------------------------------------

        f3, f4, f5 = self.neck(
            p3,
            p4,
            p5
        )

        # ----------------------------------------------------
        # Detection Head
        # ----------------------------------------------------

        outputs = self.head(
            f3,
            f4,
            f5
        )

        return outputs