import torch
import torch.nn as nn

from .blocks import ConvBlock, C3k2
from .attention import SPPF, C2PSA


class YOLO11Backbone(nn.Module):
    """
    YOLO11n-style backbone for FOD detection.

    Produces three multi-scale feature maps:

        P3 -> stride 8
        P4 -> stride 16
        P5 -> stride 32
    """

    def __init__(self):
        super().__init__()

        # ====================================================
        # Stem
        # ====================================================

        self.stem = ConvBlock(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2
        )

        # ====================================================
        # Stage 1
        #
        # 320 -> 160
        # ====================================================

        self.down1 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2
        )

        self.c3k2_1 = C3k2(
            in_channels=32,
            out_channels=32,
            num_blocks=1
        )

        # ====================================================
        # Stage 2
        #
        # 160 -> 80
        # ====================================================

        self.down2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2
        )

        self.c3k2_2 = C3k2(
            in_channels=64,
            out_channels=64,
            num_blocks=2
        )

        # ====================================================
        # Stage 3
        #
        # 80 -> 40
        # ====================================================

        self.down3 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2
        )

        self.c3k2_3 = C3k2(
            in_channels=128,
            out_channels=128,
            num_blocks=2
        )

        # ====================================================
        # Stage 4
        #
        # 40 -> 20
        # ====================================================

        self.down4 = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2
        )

        self.c3k2_4 = C3k2(
            in_channels=256,
            out_channels=256,
            num_blocks=2
        )

        # ====================================================
        # SPPF
        # ====================================================

        self.sppf = SPPF(
            in_channels=256,
            out_channels=256
        )

        # ====================================================
        # C2PSA
        # ====================================================

        self.c2psa = C2PSA(
            in_channels=256,
            out_channels=256,
            num_blocks=1,
            num_heads=4
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # 640 -> 320
        # ----------------------------------------------------

        x = self.stem(x)

        # ----------------------------------------------------
        # Stage 1
        # 320 -> 160
        # ----------------------------------------------------

        x = self.down1(x)
        x = self.c3k2_1(x)

        # ----------------------------------------------------
        # Stage 2
        # 160 -> 80
        # ----------------------------------------------------

        x = self.down2(x)
        x = self.c3k2_2(x)

        # P3 = stride 8
        p3 = x

        # ----------------------------------------------------
        # Stage 3
        # 80 -> 40
        # ----------------------------------------------------

        x = self.down3(x)
        x = self.c3k2_3(x)

        # P4 = stride 16
        p4 = x

        # ----------------------------------------------------
        # Stage 4
        # 40 -> 20
        # ----------------------------------------------------

        x = self.down4(x)
        x = self.c3k2_4(x)

        # ----------------------------------------------------
        # SPPF
        # ----------------------------------------------------

        x = self.sppf(x)

        # ----------------------------------------------------
        # C2PSA
        # ----------------------------------------------------

        x = self.c2psa(x)

        # P5 = stride 32
        p5 = x

        return p3, p4, p5