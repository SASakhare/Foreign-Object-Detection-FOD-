import torch
import torch.nn as nn

from .blocks import ConvBlock, C3k2


class YOLO11Neck(nn.Module):
    """
    YOLO11-style FPN + PAN neck.

    Input:
        P3 -> [B, 64, 80, 80]
        P4 -> [B, 128, 40, 40]
        P5 -> [B, 256, 20, 20]

    Output:
        F3 -> [B, 64, 80, 80]
        F4 -> [B, 128, 40, 40]
        F5 -> [B, 256, 20, 20]
    """

    def __init__(self):
        super().__init__()

        # ====================================================
        # TOP-DOWN FPN
        # ====================================================

        # P5: 256 -> 128
        self.reduce_p5 = ConvBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1
        )

        # P4 fusion:
        # 128 from P4 + 128 from upsampled P5
        # -> 128
        self.fpn_p4 = C3k2(
            in_channels=256,
            out_channels=128,
            num_blocks=1
        )

        # P4 -> 64
        self.reduce_p4 = ConvBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            stride=1
        )

        # P3 fusion:
        # 64 from P3 + 64 from upsampled P4
        # -> 64
        self.fpn_p3 = C3k2(
            in_channels=128,
            out_channels=64,
            num_blocks=1
        )

        # ====================================================
        # BOTTOM-UP PAN
        # ====================================================

        # P3 -> P4 resolution
        self.down_p3 = ConvBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2
        )

        # P3-down + FPN-P4
        # 64 + 128 = 192
        # -> 128
        self.pan_p4 = C3k2(
            in_channels=192,
            out_channels=128,
            num_blocks=1
        )

        # P4 -> P5 resolution
        self.down_p4 = ConvBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2
        )

        # P4-down + reduced P5
        # 128 + 128 = 256
        # -> 256
        self.pan_p5 = C3k2(
            in_channels=256,
            out_channels=256,
            num_blocks=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        p3,
        p4,
        p5
    ):

        # ====================================================
        # TOP-DOWN FPN
        # ====================================================

        # ----------------------------------------------------
        # P5 -> P4
        # ----------------------------------------------------

        p5_reduced = self.reduce_p5(
            p5
        )

        p5_up = nn.functional.interpolate(
            p5_reduced,
            scale_factor=2,
            mode="nearest"
        )

        # [B,128,40,40] + [B,128,40,40]
        p4_fused = torch.cat(
            [
                p4,
                p5_up
            ],
            dim=1
        )

        p4_fpn = self.fpn_p4(
            p4_fused
        )

        # ----------------------------------------------------
        # P4 -> P3
        # ----------------------------------------------------

        p4_reduced = self.reduce_p4(
            p4_fpn
        )

        p4_up = nn.functional.interpolate(
            p4_reduced,
            scale_factor=2,
            mode="nearest"
        )

        # [B,64,80,80] + [B,64,80,80]
        p3_fused = torch.cat(
            [
                p3,
                p4_up
            ],
            dim=1
        )

        p3_fpn = self.fpn_p3(
            p3_fused
        )

        # ====================================================
        # BOTTOM-UP PAN
        # ====================================================

        # ----------------------------------------------------
        # P3 -> P4
        # ----------------------------------------------------

        p3_down = self.down_p3(
            p3_fpn
        )

        p4_pan_input = torch.cat(
            [
                p3_down,
                p4_fpn
            ],
            dim=1
        )

        p4_pan = self.pan_p4(
            p4_pan_input
        )

        # ----------------------------------------------------
        # P4 -> P5
        # ----------------------------------------------------

        p4_down = self.down_p4(
            p4_pan
        )

        p5_pan_input = torch.cat(
            [
                p4_down,
                p5_reduced
            ],
            dim=1
        )

        p5_pan = self.pan_p5(
            p5_pan_input
        )

        # ====================================================
        # RETURN MULTI-SCALE FEATURES
        # ====================================================

        return (
            p3_fpn,
            p4_pan,
            p5_pan
        )