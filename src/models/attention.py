import torch
import torch.nn as nn

from .blocks import ConvBlock


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast.

    Applies repeated max pooling to increase
    the receptive field while keeping the
    spatial resolution unchanged.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        pool_kernel_size=5
    ):
        super().__init__()

        hidden_channels = in_channels // 2

        self.cv1 = ConvBlock(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=1,
            padding=pool_kernel_size // 2
        )

        self.cv2 = ConvBlock(
            hidden_channels * 4,
            out_channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):

        x = self.cv1(x)

        y1 = self.pool(x)

        y2 = self.pool(y1)

        y3 = self.pool(y2)

        x = torch.cat(
            [
                x,
                y1,
                y2,
                y3
            ],
            dim=1
        )

        x = self.cv2(x)

        return x





class PSABlock(nn.Module):
    """
    Position-Sensitive Attention block.

    Consists of:
        1. Multi-head self-attention
        2. Residual connection
        3. Feed-forward network
        4. Residual connection
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        expansion=0.5
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads"
            )

        self.norm1 = nn.BatchNorm2d(
            channels
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.BatchNorm2d(
            channels
        )

        hidden_channels = int(
            channels * expansion
        )

        self.ffn = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

    def forward(self, x):

        batch_size, channels, height, width = (
            x.shape
        )

        # ------------------------------------------------
        # Attention normalization
        # ------------------------------------------------

        identity = x

        x_norm = self.norm1(x)

        # ------------------------------------------------
        # Convert:
        #
        # B,C,H,W
        #
        # ->
        #
        # B,H*W,C
        # ------------------------------------------------

        x_attention = x_norm.flatten(
            2
        ).transpose(
            1,
            2
        )

        # ------------------------------------------------
        # Multi-head self attention
        # ------------------------------------------------

        attention_output, _ = (
            self.attention(
                x_attention,
                x_attention,
                x_attention
            )
        )

        # ------------------------------------------------
        # Convert back:
        #
        # B,H*W,C
        #
        # ->
        #
        # B,C,H,W
        # ------------------------------------------------

        attention_output = (
            attention_output
            .transpose(1, 2)
            .reshape(
                batch_size,
                channels,
                height,
                width
            )
        )

        # ------------------------------------------------
        # Residual connection
        # ------------------------------------------------

        x = identity + attention_output

        # ------------------------------------------------
        # FFN
        # ------------------------------------------------

        identity = x

        x_norm = self.norm2(x)

        x = self.ffn(
            x_norm
        )

        # ------------------------------------------------
        # Second residual
        # ------------------------------------------------

        x = identity + x

        return x



class C2PSA(nn.Module):
    """
    C2PSA attention block.

    Splits the input features into two branches.
    One branch is processed by PSABlock modules,
    then both branches are concatenated.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=1,
        num_heads=4,
        expansion=0.5
    ):
        super().__init__()

        hidden_channels = int(
            out_channels * expansion
        )

        # Project input
        self.cv1 = ConvBlock(
            in_channels,
            hidden_channels * 2,
            kernel_size=1,
            stride=1
        )

        # Attention blocks
        self.blocks = nn.ModuleList(
            [
                PSABlock(
                    channels=hidden_channels,
                    num_heads=num_heads
                )
                for _ in range(num_blocks)
            ]
        )

        # Final projection
        self.cv2 = ConvBlock(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):

        # -----------------------------------------------
        # Initial projection
        # -----------------------------------------------

        x = self.cv1(x)

        # -----------------------------------------------
        # Split channels
        # -----------------------------------------------

        x1, x2 = torch.chunk(
            x,
            chunks=2,
            dim=1
        )

        # -----------------------------------------------
        # Apply attention to one branch
        # -----------------------------------------------

        for block in self.blocks:

            x2 = block(x2)

        # -----------------------------------------------
        # Concatenate
        # -----------------------------------------------

        x = torch.cat(
            [
                x1,
                x2
            ],
            dim=1
        )

        # -----------------------------------------------
        # Final projection
        # -----------------------------------------------

        x = self.cv2(x)

        return x





























