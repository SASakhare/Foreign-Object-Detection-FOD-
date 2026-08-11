import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolution + BatchNorm + SiLU.

    This is the basic convolutional building block
    used throughout our YOLO11-style architecture.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1
    ):
        super().__init__()

        # Automatically calculate padding
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
    


class Bottleneck(nn.Module):
    """
    Two convolution layers with an optional
    residual/shortcut connection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5
    ):
        super().__init__()

        hidden_channels = int(
            out_channels * expansion
        )

        self.conv1 = ConvBlock(
            in_channels,
            hidden_channels,
            kernel_size=3,
            stride=1
        )

        self.conv2 = ConvBlock(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1
        )

        self.use_shortcut = (
            shortcut
            and in_channels == out_channels
        )

    def forward(self, x):

        y = self.conv1(x)

        y = self.conv2(y)

        if self.use_shortcut:
            y = y + x

        return y
    



class C3k(nn.Module):
    """
    C3-style block containing Bottleneck blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_bottlenecks=1,
        shortcut=True,
        expansion=0.5
    ):
        super().__init__()

        hidden_channels = int(
            out_channels * expansion
        )

        self.branch1 = ConvBlock(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1
        )

        self.branch2 = ConvBlock(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1
        )

        self.blocks = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    expansion=1.0
                )
                for _ in range(num_bottlenecks)
            ]
        )

        self.final_conv = ConvBlock(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):

        branch1 = self.branch1(x)

        branch2 = self.branch2(x)

        branch2 = self.blocks(
            branch2
        )

        x = torch.cat(
            [
                branch1,
                branch2
            ],
            dim=1
        )

        x = self.final_conv(x)

        return x
    




class C3k2(nn.Module):
    """
    YOLO11-style C3k2 block.

    C3k2 uses C3k blocks internally.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=1,
        shortcut=True,
        expansion=0.5
    ):
        super().__init__()

        hidden_channels = int(
            out_channels * expansion
        )

        self.cv1 = ConvBlock(
            in_channels,
            hidden_channels * 2,
            kernel_size=1,
            stride=1
        )

        self.blocks = nn.ModuleList(
            [
                C3k(
                    hidden_channels,
                    hidden_channels,
                    num_bottlenecks=1,
                    shortcut=shortcut,
                    expansion=1.0
                )
                for _ in range(num_blocks)
            ]
        )

        self.cv2 = ConvBlock(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):

        y = self.cv1(x)

        y1, y2 = torch.chunk(
            y,
            chunks=2,
            dim=1
        )

        for block in self.blocks:

            y2 = block(y2)

        y = torch.cat(
            [
                y1,
                y2
            ],
            dim=1
        )

        y = self.cv2(y)

        return y