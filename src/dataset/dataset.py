import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from ukis_pysat.raster import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A


class SegDataset(Dataset):
    def __init__(
        self,
        dataset_type: str = None,
        path: Path = None,
        transform: dict = None,
        slope: bool = False,
        s1_ratio: bool = False,
    ):
        """
        Class to load and preprocess the dataset for semantic segmentation.
        """
        self.dataset_type = dataset_type
        self.img_paths = sorted((path / "img").iterdir())
        self.mask_paths = sorted((path / "msk").iterdir())
        self.transform = transform or {}
        self.slope = slope
        self.s1_ratio = s1_ratio

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image(self.img_paths[idx], dimorder="last").arr
        mask = Image(self.mask_paths[idx], dimorder="last").arr
        mask = mask.squeeze(2) if mask.ndim == 3 else mask

        if self.dataset_type == "s2":
            rgb, nir, slope = img[..., [2, 1, 0]], img[..., 3], img[..., 4]
        elif self.dataset_type == "s1":
            vv, vh, slope = img[..., 0], img[..., 1], img[..., 2]
            if self.s1_ratio:
                epsilon = 1e-6
                ratio = (vv + vh) / (vv - vh + epsilon)
                vh = np.stack([vh, ratio], axis=-1)
                img = np.concatenate(
                    [np.expand_dims(vv, axis=-1), vh, np.expand_dims(slope, axis=-1)],
                    axis=-1,
                )
                rgb, nir = vv, vh
            else:
                rgb, nir = (
                    vv,
                    vh,
                )  # Rename the VV and Vh bands to RGB and NIR for consistency with S2

        else:
            raise ValueError("Invalid dataset type")

        if self.transform:
            img, mask = self.apply_transforms(rgb=rgb, nir=nir, slope=slope, mask=mask)
        else:
            data = ToTensorV2()(image=img, mask=mask)
            img, mask = data["image"], data["mask"]

        if self.dataset_type == "s2":

            img = img if self.slope else img[:4]
        elif self.dataset_type == "s1":
            if self.s1_ratio:
                img = img if self.slope else img[:3]
            else:
                img = img if self.slope else img[:2]

        return img, mask

    def apply_transforms(self, rgb, nir, slope, mask):
        """Applies a sequence of transformations to the input data."""
        brightness_transform = self.transform.get("brightness")
        scale_transform = self.transform.get("scale")

        # Apply brightness transform to RGB and NIR
        if brightness_transform:
            rgb, nir = self._apply_brightness_transform(rgb, nir, brightness_transform)

        # Apply scale/flip/crop transform to RGB, NIR, slope, and mask
        if scale_transform:
            rgb, nir, slope, mask = self._apply_scale_transform(
                rgb, nir, slope, mask, scale_transform
            )

        img = torch.cat((rgb, nir, slope), dim=0)
        return img, mask

    @staticmethod
    def _apply_brightness_transform(rgb, nir, brightness_transform):
        """Helper method to apply brightness transformations."""
        brightness_data = brightness_transform(image=rgb)
        rgb = brightness_data["image"]
        nir = A.ReplayCompose.replay(brightness_data["replay"], image=nir)["image"]
        return rgb, nir

    @staticmethod
    def _apply_scale_transform(rgb, nir, slope, mask, scale_transform):
        """Helper method to apply scaling transformations."""
        scale_data = scale_transform(image=rgb, mask=mask)
        rgb = scale_data["image"]
        mask = scale_data["mask"]
        nir = A.ReplayCompose.replay(scale_data["replay"], image=nir)["image"]
        slope = (
            A.ReplayCompose.replay(scale_data["replay"], image=slope)["image"]
            if slope is not None
            else None
        )
        return rgb, nir, slope, mask


def get_transform(resize=256):
    """Returns predefined transformations for data augmentation."""
    return {
        "brightness": A.ReplayCompose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                )
            ]
        ),
        "scale": A.ReplayCompose(
            [
                A.RandomScale(scale_limit=(0.9, 1.1), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=resize, width=resize, always_apply=True),
                ToTensorV2(),
            ]
        ),
    }


def get_dataset_subset(dataset, num_samples):
    indices = get_subset_indices(dataset, num_samples)
    return get_subset(dataset, indices)


def get_subset_indices(dataset, num_samples):
    return np.random.choice(len(dataset), num_samples, replace=False)


def get_subset(dataset, indices):
    return Subset(dataset, indices)


def get_dataloader(
    dataset_type=None,
    dataset_path=None,
    transform=None,
    slope=False,
    s1_ratio=False,
    subset=None,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
):

    dataset = SegDataset(
        dataset_type, dataset_path, transform=transform, slope=slope, s1_ratio=s1_ratio
    )

    if subset is not None:

        assert 0 < subset <= 100, "Subset must be a number between 0 and 100"
        selected_size = round(subset * len(dataset) // 100)
        dataset = get_dataset_subset(dataset, selected_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader
