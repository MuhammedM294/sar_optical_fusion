import numpy as np
import torch
import torch.nn as nn
import torchbnn as bnn


class BaysianDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.double_conv = nn.Sequential(
            bnn.BayesConv2d(
                prior_mu=0,
                prior_sigma=0.1,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            bnn.BayesBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            bnn.BayesConv2d(
                prior_mu=0,
                prior_sigma=0.1,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            bnn.BayesBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if dropout:
            self.double_conv.append(nn.Dropout2d(p=dropout))


if __name__ == "__main__":
    model = BaysianDoubleConv(3, 3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
    print(y)
