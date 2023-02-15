"""Unet."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vit import ViT


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """Double convolution module."""
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_13 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1
        )
        self.conv_15 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=5, padding=2
        )
        self.batch_norm1 = nn.BatchNorm2d(2 * mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv_23 = nn.Conv2d(
            2 * mid_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv_25 = nn.Conv2d(
            2 * mid_channels, out_channels, kernel_size=5, padding=2
        )
        self.batch_norm2 = nn.BatchNorm2d(2 * out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass."""
        out1 = self.conv_13(x)
        out2 = self.conv_15(x)
        # do concatenation store result in out
        out = torch.cat((out1, out2), dim=1)
        out = self.batch_norm1(out)
        out = self.relu1(out)

        out1 = self.conv_23(out)
        out2 = self.conv_25(out)
        # do concatenation store result in out
        out = torch.cat((out1, out2), dim=1)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool followed by DoubleConv."""

    def __init__(self, in_channels, out_channels):
        """Downscale."""
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Forward pass."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling and then DoubleConv."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """Upscale."""
        super().__init__()

        # if bilinear, use the normal convolutions
        # to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass."""
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Unet final convolution module."""

    def __init__(self, in_channels, out_channels):
        """Unet output convolution."""
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass."""
        return self.conv(x)


class UNet(nn.Module):
    """Unet."""

    def __init__(self, n_channels, n_classes, bilinear=True):
        """Unet module."""
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(64, 64)
        self.down2 = Down(128, 128)
        self.down3 = Down(256, 256)
        factor = 2 if bilinear else 1

        self.vit = ViT(
            image_size=64,
            patch_size=8,
            dim=2048,
            depth=2,
            heads=16,
            mlp_dim=12,
            channels=512,
        )  # dim%head=0
        self.vit_conv = nn.Conv2d(
            32, 512, kernel_size=1, padding=0
        )  # to increase the number of channels
        self.vit_linear = nn.Linear(64, 512)

        self.up1 = Up(512, 128 // factor, bilinear)
        self.up2 = Up(256, 64 // factor, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """Forward pass."""
        # down sizing
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # applying Vision Transformer for feature extraction
        x5 = self.vit(x4)
        x5 = torch.reshape(x5, (-1, 32, 8, 8))
        x6 = self.vit_conv(x5)
        x7 = self.vit_linear(torch.reshape(x6, (-1, 512, 64)))
        x8 = torch.reshape(x7, (-1, 256, 32, 32))

        # up scaling
        x = self.up1(x8, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits
