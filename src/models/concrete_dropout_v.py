import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcreteDropout2d(nn.Module):
    """
    Concrete Dropout for 2D convolutional feature maps.
    - Learns dropout probability p via a trainable logit parameter.
    - Applies channel-wise (feature-map) dropout (like Dropout2d).
    - Computes regularization term for the layer (to be added to the main loss).
    """

    def __init__(
        self,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5,
        init_min_p: float = 0.1,
        temp: float = 0.1,
        apply_in_eval: bool = True,
    ):
        """
        Args:
            weight_regularizer: λ (scales ||W||^2 term)
            dropout_regularizer: scale for the entropy term
            init_min_p: initial dropout prob (around)
            temp: temperature for Concrete distribution
            apply_in_eval: if True, still sample masks in eval() (MC Dropout); else deterministic in eval
        """
        super().__init__()
        # initialize logit so sigmoid(logit) ≈ init_min_p
        init_logit = torch.log(
            torch.tensor(init_min_p) / (1.0 - torch.tensor(init_min_p))
        )
        self.logit_p = nn.Parameter(init_logit)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.temp = temp
        self.eps = 1e-7
        self.apply_in_eval = apply_in_eval

        # last computed regularization (scalar tensor)
        self.regularization = torch.tensor(0.0, device=self.logit_p.device)

    def forward(self, x: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, C, H, W)
            layer: the convolutional layer whose weights are regularized
        Returns:
            out: result of layer applied to (dropped + rescaled) x
        Side-effect:
            sets self.regularization to a scalar tensor (to be summed to the overall reg loss)
        """
        p = torch.sigmoid(self.logit_p)  # scalar in (0,1)
        if (not self.training) and (not self.apply_in_eval):
            # deterministic path: no dropout sampling (like inference without MC)
            out = layer(x)
            # regularization still computed and returned so training can use it if needed
            weight_reg = (
                self.weight_regularizer
                * torch.sum(layer.weight**2)
                / (1.0 - p + self.eps)
            )
            dropout_reg = p * torch.log(p + self.eps) + (1.0 - p) * torch.log(
                1.0 - p + self.eps
            )
            # scale dropout regularizer by number of channels (feature maps) to normalize magnitude
            dropout_reg = dropout_reg * self.dropout_regularizer * x.shape[1]
            self.regularization = weight_reg + dropout_reg
            return out

        # Concrete distribution sampling (channel-wise: one random per (B, C, 1, 1))
        batch, channels, _, _ = x.shape
        device = x.device

        # Sample u with shape (B, C, 1, 1) then broadcast; avoids per-pixel noisy masks
        u = torch.rand((batch, channels, 1, 1), device=device)

        # Concrete / Gumbel-Softmax trick: compute continuous relaxation of Bernoulli
        # logit(p) + log u - log(1-u)  then sigmoid(temp^-1 * ...)
        logit_p = torch.log(p + self.eps) - torch.log(1.0 - p + self.eps)
        random_logit = logit_p + torch.log(u + self.eps) - torch.log(1.0 - u + self.eps)
        drop_prob = torch.sigmoid(
            random_logit / self.temp
        )  # in (0,1), shape (B, C, 1, 1)
        retain_prob = 1.0 - drop_prob  # stochastic retain

        # Apply mask and rescale by expected keep prob (1-p)
        x_dropped = x * retain_prob
        x_dropped = x_dropped / (1.0 - p + self.eps)

        out = layer(x_dropped)

        # Regularization terms
        # Weight regularization (encourages smaller weights, scaled by (1-p) as per paper)
        weight_reg = (
            self.weight_regularizer * torch.sum(layer.weight**2) / (1.0 - p + self.eps)
        )

        # Dropout regularization (entropy-like term). Keep sign positive so reg is positive.
        dropout_reg = p * torch.log(p + self.eps) + (1.0 - p) * torch.log(
            1.0 - p + self.eps
        )
        # scale by number of feature maps (channels) to give reasonable magnitude
        dropout_reg = dropout_reg * self.dropout_regularizer * x.shape[1]

        self.regularization = weight_reg + dropout_reg

        return out


class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2 with optional Concrete Dropout per conv."""

    def __init__(self, in_channels, out_channels, use_concrete_dropout: bool = True):
        super().__init__()
        # conv layers (bias=False because we use BatchNorm)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_concrete_dropout = use_concrete_dropout
        if use_concrete_dropout:
            # Provide reasonable defaults; can be tuned externally
            self.cd1 = ConcreteDropout2d()
            self.cd2 = ConcreteDropout2d()
        else:
            self.dropout = nn.Dropout2d(0.5)

        # Sum of regs from the two convs (kept as tensor for safe device handling)
        self.regularization = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_concrete_dropout:
            # cd.forward returns conv(x_dropped)
            x = F.relu(self.bn1(self.cd1(x, self.conv1)))
            x = F.relu(self.bn2(self.cd2(x, self.conv2)))
            # accumulate regularization (ensure tensor)
            self.regularization = self.cd1.regularization + self.cd2.regularization
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            self.regularization = torch.tensor(0.0, device=x.device)
        return x


class EncoderBlock(nn.Module):
    """Downsampling block: MaxPool + DoubleConv."""

    def __init__(self, in_channels, out_channels, use_concrete_dropout: bool = True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(
            in_channels, out_channels, use_concrete_dropout=use_concrete_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class DecoderBlock(nn.Module):
    """Upsampling block: ConvTranspose2d + DoubleConv.
    Note: in_channels here corresponds to channels of the up-input (typically after bottleneck),
    and DoubleConv is constructed expecting concatenated channels (up + skip).
    """

    def __init__(self, in_channels, out_channels, use_concrete_dropout: bool = True):
        super().__init__()
        # upconvolution reduces in_channels -> out_channels for spatial upsample
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After upsample we will concat with skip (so DoubleConv should accept 2*out_channels as input)
        self.double_conv = DoubleConv(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            use_concrete_dropout=use_concrete_dropout,
        )

    def forward(self, x: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # if shapes mismatch due to odd sizes, interpolate to match skip spatial dims
        if x.shape[-2:] != skip_features.shape[-2:]:
            x = F.interpolate(
                x, size=skip_features.shape[-2:], mode="bilinear", align_corners=True
            )
        # concat along channel dim
        x = torch.cat([x, skip_features], dim=1)
        x = self.double_conv(x)
        return x


class UNetConcrete(nn.Module):
    """U-Net with Concrete Dropout in convolutional blocks.
    Returns (output, reg_loss) from forward.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        features=(64, 128, 256, 512),
        use_concrete_dropout=True,
    ):
        super().__init__()
        self.use_concrete_dropout = use_concrete_dropout

        # Initial conv (no pooling before it)
        self.init_conv = DoubleConv(
            in_channels, features[0], use_concrete_dropout=use_concrete_dropout
        )

        # Encoder (downsampling stack)
        self.encoder_blocks = nn.ModuleList()
        for i in range(1, len(features)):
            self.encoder_blocks.append(
                EncoderBlock(
                    features[i - 1],
                    features[i],
                    use_concrete_dropout=use_concrete_dropout,
                )
            )

        # Bottleneck (no pool here; typically after final encoder block)
        self.center = DoubleConv(
            features[-1], features[-1] * 2, use_concrete_dropout=use_concrete_dropout
        )

        # Decoder (upsampling stack)
        self.decoder_blocks = nn.ModuleList()
        for feature in reversed(features):
            # Expect in_channels to the upconv equal to center/out channels appropriately:
            # we use ConvTranspose2d(in_channels, out_channels) with in_channels = feature*2, out_channels=feature
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=feature * 2,
                    out_channels=feature,
                    use_concrete_dropout=use_concrete_dropout,
                )
            )

        # Final segmentation head (maps to out_channels)
        self.segmentation_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skip_connections = []
        # Keep total reg loss as tensor on correct device
        reg_loss = torch.tensor(0.0, device=x.device)

        # Encoder path
        x = self.init_conv(x)
        skip_connections.append(x)
        reg_loss = reg_loss + getattr(
            self.init_conv, "regularization", torch.tensor(0.0, device=x.device)
        )

        for enc in self.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)
            reg_loss = reg_loss + getattr(
                enc.double_conv, "regularization", torch.tensor(0.0, device=x.device)
            )

        # Bottleneck
        x = self.center(x)
        reg_loss = reg_loss + getattr(
            self.center, "regularization", torch.tensor(0.0, device=x.device)
        )

        # Decoder path: reversed skip connections (matching spatial resolution)
        skip_connections = skip_connections[::-1]
        for i, dec in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            x = dec(x, skip)
            reg_loss = reg_loss + getattr(
                dec.double_conv, "regularization", torch.tensor(0.0, device=x.device)
            )

        out = self.segmentation_head(x)
        return out, reg_loss


# -------------------------
# 🔍 Quick test
# -------------------------
if __name__ == "__main__":
    # small smoke test
    model = UNetConcrete(
        in_channels=3,
        out_channels=2,
        features=(64, 128, 256, 512),
        use_concrete_dropout=True,
    )
    model.train()
    x = torch.randn(2, 3, 128, 128)  # batch of 2
    y, reg_loss = model(x)
    print("Output shape:", y.shape)
    print("Regularization loss:", float(reg_loss))

    # Example of computing final loss:
    # criterion = nn.CrossEntropyLoss()
    # target = torch.randint(0, 2, (2, 128, 128), dtype=torch.long)
    # loss = criterion(y, target) + reg_loss
    # loss.backward()
