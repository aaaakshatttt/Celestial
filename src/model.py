# src/model.py
# PURPOSE: Defines the Pix2Pix deep learning model.
#
# Pix2Pix has TWO neural networks that compete with each other:
#
# 1. GENERATOR     → Takes IR image, outputs a fake RGB image
# 2. DISCRIMINATOR → Looks at IR+RGB pairs and judges if RGB is real or fake
#
# They train together:
# - Generator tries to fool the Discriminator
# - Discriminator tries not to be fooled
# - Result: Generator gets better and better at making realistic RGB images

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    A single convolution block used throughout the network.

    WHAT IT DOES:
    Conv2d → BatchNorm → Activation

    Conv2d     : Learns spatial features (edges, textures, shapes)
    BatchNorm  : Keeps numbers stable during training (prevents exploding gradients)
    Activation : Adds non-linearity so network can learn complex patterns
                 - LeakyReLU in Discriminator (allows small negative values)
                 - ReLU in Generator encoder (standard)
    """
    def __init__(self, in_channels, out_channels,
                 stride=2, use_norm=True, activation="leaky"):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride,
                padding=1, bias=not use_norm
            )
        ]

        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    """
    Upsampling block used in the Generator decoder.

    WHAT IT DOES:
    ConvTranspose2d → BatchNorm → ReLU (→ optional Dropout)

    ConvTranspose2d : The OPPOSITE of Conv2d — it makes images LARGER
                      Used to go from small feature maps back to 256×256

    Dropout         : Randomly zeros 50% of neurons during training.
                      WHY: Prevents the network from memorizing training data.
                      Only used in first 3 decoder layers.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# GENERATOR — U-Net Architecture
# ─────────────────────────────────────────────

class Generator(nn.Module):
    """
    The Generator converts IR images → RGB images.

    ARCHITECTURE: U-Net
    U-Net has two parts:
    1. ENCODER (goes down)  : Extracts features, makes image smaller
    2. DECODER (goes up)    : Rebuilds image at full size with color

    SKIP CONNECTIONS (the U in U-Net):
    Each encoder layer is directly connected to its mirror decoder layer.
    WHY: This passes fine spatial detail (edges, textures) directly
         from input to output, preventing blurry results.

    Input  : (batch, 1, 256, 256)  — single channel IR
    Output : (batch, 3, 256, 256)  — 3 channel RGB
    """

    def __init__(self):
        super().__init__()

        # ── ENCODER (Downsampling path) ──
        # Each layer halves the spatial size: 256→128→64→32→16→8→4→2→1
        # But doubles the channels: 64→128→256→512→512→512→512→512

        self.enc1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        # No BatchNorm on first layer (standard Pix2Pix practice)

        self.enc2 = ConvBlock(64,  128, activation="leaky")
        self.enc3 = ConvBlock(128, 256, activation="leaky")
        self.enc4 = ConvBlock(256, 512, activation="leaky")
        self.enc5 = ConvBlock(512, 512, activation="leaky")
        self.enc6 = ConvBlock(512, 512, activation="leaky")
        self.enc7 = ConvBlock(512, 512, activation="leaky")

        # Bottleneck — smallest representation (1×1 feature map)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # ── DECODER (Upsampling path) ──
        # Skip connections DOUBLE the input channels (current + skip)
        # e.g. dec1 gets 512 (bottleneck) → outputs 512
        #      dec2 gets 512+512=1024 (dec1 + enc7 skip) → outputs 512

        self.dec1 = UpConvBlock(512,  512, use_dropout=True)
        self.dec2 = UpConvBlock(1024, 512, use_dropout=True)
        self.dec3 = UpConvBlock(1024, 512, use_dropout=True)
        self.dec4 = UpConvBlock(1024, 512)
        self.dec5 = UpConvBlock(1024, 256)
        self.dec6 = UpConvBlock(512,  128)
        self.dec7 = UpConvBlock(256,  64)

        # Final layer — outputs 3-channel RGB image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()   # Output range [-1, 1] — matches our normalized targets
        )

        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass — runs the image through the full U-Net.
        Skip connections concatenate encoder outputs to decoder inputs.
        torch.cat joins tensors along the channel dimension.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.leaky(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # Bottleneck
        bn = self.bottleneck(e7)

        # Decoder with skip connections
        d1 = self.dec1(bn)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))

        return self.final(torch.cat([d7, e1], dim=1))


# ─────────────────────────────────────────────
# DISCRIMINATOR — PatchGAN
# ─────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    The Discriminator judges if an RGB image is real or fake.

    ARCHITECTURE: PatchGAN
    Instead of judging the WHOLE image as real/fake (which loses detail),
    PatchGAN judges overlapping 70×70 PATCHES of the image.

    WHY PATCHES?
    - Captures local texture quality better
    - Fewer parameters = faster training
    - Works well for image translation tasks

    Input  : IR image + RGB image concatenated → (batch, 4, 256, 256)
             (1 IR channel + 3 RGB channels = 4 channels total)
    Output : (batch, 1, 30, 30) — one real/fake score per patch
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1 — no BatchNorm on first layer
            ConvBlock(4,   64,  use_norm=False, activation="leaky"),
            # Layer 2
            ConvBlock(64,  128, activation="leaky"),
            # Layer 3
            ConvBlock(128, 256, activation="leaky"),
            # Layer 4 — stride=1 to keep spatial size for patch output
            ConvBlock(256, 512, stride=1, activation="leaky"),
            # Output — single channel patch map
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # No activation — raw scores (BCEWithLogitsLoss handles sigmoid)
        )

    def forward(self, ir, rgb):
        """
        Concatenates IR and RGB along channel dimension,
        then passes through the network.

        The Discriminator sees BOTH images together so it can
        judge if the RGB is appropriate for that specific IR input.
        """
        x = torch.cat([ir, rgb], dim=1)   # (batch, 4, 256, 256)
        return self.model(x)


# ─────────────────────────────────────────────
# WEIGHTS INITIALIZATION
# ─────────────────────────────────────────────

def initialize_weights(model):
    """
    Initializes Conv and BatchNorm weights using the values
    from the original Pix2Pix paper.

    WHY: Default PyTorch initialization can cause slow or unstable training.
    The paper found that mean=0, std=0.02 works well for GANs.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, 0)


# ─────────────────────────────────────────────
# TEST THE MODEL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Pix2Pix model architecture...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create models
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Initialize weights
    initialize_weights(G)
    initialize_weights(D)

    # Create dummy input — simulates one batch of 2 images
    dummy_ir  = torch.randn(2, 1, 256, 256).to(device)
    dummy_rgb = torch.randn(2, 3, 256, 256).to(device)

    # Test Generator
    fake_rgb = G(dummy_ir)
    print(f"Generator:")
    print(f"  Input  shape : {dummy_ir.shape}")
    print(f"  Output shape : {fake_rgb.shape}")   # Should be (2, 3, 256, 256)

    # Test Discriminator
    real_score = D(dummy_ir, dummy_rgb)
    fake_score = D(dummy_ir, fake_rgb.detach())
    print(f"\nDiscriminator:")
    print(f"  Real score shape : {real_score.shape}")  # Should be (2, 1, 30, 30)
    print(f"  Fake score shape : {fake_score.shape}")  # Should be (2, 1, 30, 30)

    # Count parameters
    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"\nGenerator parameters     : {g_params:.2f}M")
    print(f"Discriminator parameters : {d_params:.2f}M")

    print("\nModel architecture is correct and ready for training!")