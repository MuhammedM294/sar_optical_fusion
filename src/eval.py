import os
import sys
import torch
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset.dataset import SegDataset
from dataset.utils import get_transform
from train import DATA_MEANS, DATA_STD, get_device
from src.models.unet import UNet

TEST_PATH = Path("data/patches/val")
model_path = Path("models/model_25.pth")
test_dataset = SegDataset(
    TEST_PATH,
    transform=get_transform(is_train=False, mean=DATA_MEANS, std=DATA_STD),
    resize=None,
)


if __name__ == "__main__":

    # Visual Inspection
    # Load model
    device = get_device()
    model = UNet(in_channels=3, out_channels=1).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    rand_idx = torch.randint(0, len(test_dataset), (1,)).item()
    print(f"Random index: {rand_idx}")
    # Evaluate model
    img, mask = test_dataset[rand_idx]
    img = img.to(device, dtype=torch.float32)

    # Add batch dimension
    img = img.unsqueeze(0)
    pred = model(img)

    # Denormalize image
    mean = torch.tensor(DATA_MEANS).view(1, 3, 1, 1).to(device)
    std = torch.tensor(DATA_STD).view(1, 3, 1, 1).to(device)

    img = img * std + mean
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()
    pred = torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().detach().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")
    plt.show()
