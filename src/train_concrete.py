import os
import sys
import random
import torch
import wandb
import torchmetrics
import numpy as np
from pathlib import Path

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, DiceLoss
import torchmetrics.classification
from tqdm import tqdm
import torch.nn.functional as F
from src.dataset.dataset import get_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_deterministic_behavior():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def criterion(logits, mask, mode="binary", pos_weight=None):

    # Binary Cross-Entropy with class weighting for imbalance
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask)

    dice_loss = DiceLoss(mode=mode)(logits, mask)
    # lovasz_loss = LovaszLoss(mode=mode)(logits, mask)
    # combined_loss = 0.5 * bce_loss + 0.5 * lovasz_loss
    combined_loss = bce_loss + dice_loss
    return combined_loss


class IoU(torchmetrics.Metric):
    def __init__(self, num_classes, include_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # Ensure the inputs are boolean
        preds = preds.to(torch.bool)
        target = target.to(torch.bool)

        intersection = torch.sum((preds & target))  # Logical AND
        union = torch.sum((preds | target))  # Logical OR

        self.iou_sum += intersection / (union + 1e-6)
        self.count += 1

    def compute(self):
        return self.iou_sum / self.count


def get_metrics(task="binary", num_classes=1):

    return torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.classification.Accuracy(
                task=task, num_classes=num_classes
            ),
            "precision": torchmetrics.classification.Precision(
                task=task, num_classes=num_classes
            ),
            "recall": torchmetrics.classification.Recall(
                task=task, num_classes=num_classes
            ),
            "f1": torchmetrics.classification.F1Score(
                task=task, num_classes=num_classes
            ),
            "iou": IoU(
                num_classes=num_classes,
                include_background=False,
            ),
        }
    )


def log_metrics(epoch, epoch_loss, metrics_values, metric_type, writer, logger):
    """
    Logs metrics to TensorBoard and the console for training or validation.
    """
    # Log metrics to TensorBoard
    for key, value in metrics_values.items():
        writer.add_scalar(
            f"{metric_type}/{key.capitalize()}", value.item() * 100, epoch + 1
        )

    # Create a dictionary with rounded metrics
    metrics_dict = {
        key.capitalize(): round(value.item() * 100, 2)
        for key, value in metrics_values.items()
    }

    # Format the log message
    log_message = (
        f"{metric_type[0]}-Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | "
        f"Accuracy: {metrics_dict.get('Accuracy', 'N/A')} | "
        f"Precision: {metrics_dict.get('Precision', 'N/A')} | "
        f"Recall: {metrics_dict.get('Recall', 'N/A')} | "
        f"F1: {metrics_dict.get('F1', 'N/A')} | "
        f"IoU: {metrics_dict.get('Iou', 'N/A')}"
    )

    # Log to the console
    logger.info(log_message)


def train(
    model,
    base_path,
    dataset_type,
    train_path,
    train_data_opt,
    val_path,
    val_data_opt,
    optimizer,
    criterion,
    train_metrics,
    val_metrics,
    num_epochs,
    grad_scaler,
    device,
    scheduler,
    early_stopping_patience,
    logger,
    tensorboard_logdir,  # Path for TensorBoard logs
):
    # TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=tensorboard_logdir)

    train_loader = get_dataloader(
        dataset_type,
        train_path,
        **train_data_opt,
    )

    val_loader = get_dataloader(
        dataset_type,
        val_path,
        **val_data_opt,
    )

    train_metrics.to(device)
    val_metrics.to(device)

    logger.info("Start Training ...")
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        with tqdm(
            total=len(train_loader),
            desc=f"Train Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for img, mask in train_loader:
                # Move data to device
                img = img.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                logits, reg_loss = model(img)
                mask = mask.unsqueeze(1)

                # Calculate loss
                loss = criterion(logits, mask) + reg_loss
                prediction = (torch.sigmoid(logits) > 0.5).float()

                # Backward pass and optimization
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_train_loss += loss.item()

                # Log metrics for this batch
                writer.add_scalar(
                    "Train/Loss_per_batch",
                    loss.item(),
                    epoch * len(train_loader) + pbar.n,
                )

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "acc": train_metrics["accuracy"](prediction, mask).item(),
                        "prec": train_metrics["precision"](prediction, mask).item(),
                        "reca": train_metrics["recall"](prediction, mask).item(),
                        "f1": train_metrics["f1"](prediction, mask).item(),
                        "iou": train_metrics["iou"](prediction, mask).item(),
                    }
                )
                pbar.update()

            epoch_train_loss /= len(train_loader)
            train_metrics_values = train_metrics.compute()
            log_metrics(
                epoch, epoch_train_loss, train_metrics_values, "train", writer, logger
            )

            train_metrics.reset()

        torch.cuda.empty_cache()
        model.eval()

        with torch.no_grad():
            epoch_val_loss = 0.0

            with tqdm(
                total=len(val_loader),
                desc=f"Validation Epoch {epoch + 1}/{num_epochs}",
                unit="batch",
            ) as pbar:
                for img, mask in val_loader:
                    img = img.to(device, dtype=torch.float32)
                    mask = mask.to(device, dtype=torch.float32)
                    mask = mask.unsqueeze(1)

                    logits, reg_loss = model(img)
                    loss = criterion(logits, mask) + reg_loss
                    prediction = (torch.sigmoid(logits) > 0.5).float()

                    epoch_val_loss += loss.item()

                    # Log metrics for this batch
                    writer.add_scalar(
                        "Validation/Loss_per_batch",
                        loss.item(),
                        epoch * len(val_loader) + pbar.n,
                    )

                    pbar.set_postfix(
                        {
                            "loss": loss.item(),
                            "acc": val_metrics["accuracy"](prediction, mask).item(),
                            "prec": val_metrics["precision"](prediction, mask).item(),
                            "reca": val_metrics["recall"](prediction, mask).item(),
                            "f1": val_metrics["f1"](prediction, mask).item(),
                            "iou": val_metrics["iou"](prediction, mask).item(),
                        }
                    )
                    pbar.update()

            epoch_val_loss /= len(val_loader)
            val_metrics_values = val_metrics.compute()
            log_metrics(
                epoch, epoch_val_loss, val_metrics_values, "validation", writer, logger
            )
            val_metrics.reset()

        # Log epoch-wise loss
        writer.add_scalar("Train/Loss", epoch_train_loss, epoch + 1)
        writer.add_scalar("Validation/Loss", epoch_val_loss, epoch + 1)

        # Scheduler step and early stopping logic
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0

            # Save the best model
            model_filename = f"{train_path.parts[-3]}_{train_path.parts[-2]}"
            model_filename += "_slope_" if train_data_opt["slope"] else ""
            model_filename += "_ratio_" if train_data_opt["s1_ratio"] else ""
            model_filename += model.__class__.__name__
            best_model_path = base_path / f"models/{model_filename}_concrete_v2.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stopping_patience:
            logger.info("Early stopping triggered.")
            break

    writer.close()
    logger.info("Training complete.")
