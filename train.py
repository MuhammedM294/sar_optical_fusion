import argparse
import torch
from pathlib import Path
from torch import optim
from src.logger.logger import create_logger
from src.dataset.dataset import get_transform
from src.models.unet import UNet
from src.models.concrete_dropout_v import UNetConcrete
from src.models.unet_plus import UNetPlusPlus
from src.train import (
    set_seed,
    configure_deterministic_behavior,
    get_device,
    criterion,
    train,
    get_metrics,
)

from src.train_concrete import train as train_concrete


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="s2",
        choices=["s2", "s1"],
        help="Dataset type (s1 or s2)",
        required=True,
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="linear",
        choices=["zscore", "linear"],
        help="Normalization type (zscore or linear minmax)",
        required=True,
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        choices=[256, 512],
        help="Image size (256 or 512)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "unet++", "concrete"],
        help="Model architecture (unet or unet++)",
        required=True,
    )
    parser.add_argument(
        "--slope",
        action="store_true",
        help="Include slope data in the input",
    )
    parser.add_argument(
        "--s1_ratio",
        action="store_true",
        help="Include s1 ratio band in the input",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    DIR_BASE = Path("/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood")
    TRAIN_PATH = (
        DIR_BASE / f"data/patches/{args.dataset}/{args.norm_type}_{args.resize}/train"
    )
    VAL_PATH = (
        DIR_BASE / f"data/patches/{args.dataset}/{args.norm_type}_{args.resize}/val"
    )
    TENSORBOARD_LOGDIR = DIR_BASE / "logs/events/"

    # Logger setup
    train_logger_name = (
        f"train_{args.dataset}_{args.norm_type}_{args.resize}_{args.model}"
    )
    logger = create_logger(train_logger_name)
    train_logger_file = DIR_BASE / f"logs/{train_logger_name}.log"
    assert train_logger_file.exists(), f"Logger file {train_logger_file} does not exist"

    batch_size = 64 if args.resize == 256 else 16
    # Dataset options
    DATA_OPT = {
        "transform": get_transform(args.resize),
        "slope": args.slope,
        "s1_ratio": args.s1_ratio,
        "subset": None,  # Use the entire dataset
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": False,
    }

    # Training hyperparameters
    HYPERPARAMS = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "beta1": 0.9,
        "beta2": 0.9,
        "epochs": 100,
        "early_stop_patience": 15,
        "lr_reduce_patience": 5,
        "lr_reduce_factor": 0.1,
    }

    # Set random seed and enable deterministic behavior
    set_seed(42)
    configure_deterministic_behavior()

    # Device configuration
    device = get_device()
    if args.dataset == "s2":
        in_channels = 5 if DATA_OPT["slope"] else 4
        DATA_OPT["s1_ratio"] = False
    elif args.dataset == "s1":
        if DATA_OPT["s1_ratio"]:
            in_channels = 4 if DATA_OPT["slope"] else 3
        else:
            in_channels = 3 if DATA_OPT["slope"] else 2
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    # Initialize model
    if args.model == "unet":
        model = UNet(in_channels=in_channels, out_channels=1).to(device)

        # # this is a temporary fix to load the model to continue training
        # model_path = Path(
        #     "/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood/models/s1_zscore_256_slope_UNet.pth"
        # )
        # model.load_state_dict(torch.load(model_path))
    elif args.model == "unet++":
        model = UNetPlusPlus(
            in_channels=in_channels, out_channels=1, deep_supervision=False
        ).to(device)

        # this is a temporary fix to load the model to continue training
        # model_path = Path(
        #     "/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood/models/best_s1_zscore_256_slope_UNetPlusPlus.pth"
        # )
        # model.load_state_dict(torch.load(model_path))

    elif args.model == "concrete":
        model = UNetConcrete(in_channels=in_channels, out_channels=1).to(device)

        # # this is a temporary fix to load the model to continue training
        # print(f"Loading concrete model weights for {args.dataset} concrete model...")

        # model_path = Path(
        #     f"/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood/models/s1_zscore_256_slope_UNetConcrete_concrete_v2 copy.pth"
        # )
        # model.load_state_dict(torch.load(model_path))

    # Optimizer and scheduler setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"],
        betas=(HYPERPARAMS["beta1"], HYPERPARAMS["beta2"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=HYPERPARAMS["lr_reduce_patience"],
        factor=HYPERPARAMS["lr_reduce_factor"],
    )

    # Gradient scaler for mixed precision training
    grad_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Metrics setup
    train_metrics = get_metrics()
    val_metrics = get_metrics()

    # Data options for training and validation
    train_data_opt = DATA_OPT.copy()
    val_data_opt = DATA_OPT.copy()
    val_data_opt["transform"] = {}  # No augmentation for validation

    if args.model != "concrete":
        train_funct = train
    else:
        train_funct = train_concrete
    # Start training
    train_funct(
        model=model,
        base_path=DIR_BASE,
        dataset_type=args.dataset,
        train_path=TRAIN_PATH,
        train_data_opt=train_data_opt,
        val_path=VAL_PATH,
        val_data_opt=val_data_opt,
        optimizer=optimizer,
        criterion=criterion,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        num_epochs=HYPERPARAMS["epochs"],
        grad_scaler=grad_scaler,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=HYPERPARAMS["early_stop_patience"],
        logger=logger,
        tensorboard_logdir=TENSORBOARD_LOGDIR,
    )
