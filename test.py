from pathlib import Path
import torch
from tqdm import tqdm
from torch import optim
from src.logger.logger import create_logger
from src.dataset.dataset import get_dataloader
from src.models.unet import UNet
from src.models.unet_plus import UNetPlusPlus
from src.dataset.dataset import SegDataset
from src.models.concrete_dropout import UNet as UNetConcrete
from src.train import (
    criterion,
    get_metrics,
    get_device,
    set_seed,
    configure_deterministic_behavior,
)

DIR_BASE = Path("/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood")
TEST_PATH = DIR_BASE / f"data/patches/"
MODELS_PATH = [
    model
    for model in Path(DIR_BASE / "models").rglob("*.pth")
    if not model.name.startswith(".")
]
VAL_OR_TEST = "test"
logger = create_logger(VAL_OR_TEST)
test_logger_file = DIR_BASE / f"logs/{VAL_OR_TEST}.log"
assert test_logger_file.exists(), f"Logger file {test_logger_file} does not exist"


def test(
    model,
    test_path,
    test_opt,
    logger,
    device,
    model_path,
    test_metrics,
    val_or_test="test",
):
    model_path_names = model_path.name.split("_")
    model_path_names[-1] = model_path_names[-1].split(".")[0]
    test_path = (
        test_path
        / model_path_names[0]
        / f"{model_path_names[1]}_{model_path_names[2]}"
        / f"{val_or_test}"
    )

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    test_loader = get_dataloader(
        model_path_names[0],
        test_path,
        **test_opt,
    )
    test_metrics.to(device)
    logger.info(f"Model {model_path.name} ...")
    with torch.no_grad():
        test_loss = 0.0

        with tqdm(
            total=len(test_loader),
            desc=f"Test",
            unit="batch",
        ) as pbar:
            for img, mask in test_loader:
                img = img.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.float32)
                mask = mask.unsqueeze(1)
                if isinstance(model, UNetConcrete):
                    logits, reg_loss = model(img)
                    loss = criterion(logits, mask) + reg_loss
                else:
                    logits = model(img)
                    loss = criterion(logits, mask)
                prediction = (torch.sigmoid(logits) > 0.5).float()

                test_loss += loss.item()

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "acc": test_metrics["accuracy"](prediction, mask).item(),
                        "prec": test_metrics["precision"](prediction, mask).item(),
                        "reca": test_metrics["recall"](prediction, mask).item(),
                        "f1": test_metrics["f1"](prediction, mask).item(),
                        "iou": test_metrics["iou"](prediction, mask).item(),
                    }
                )
                pbar.update()

        test_loss /= len(test_loader)
        test_metrics_values = test_metrics.compute()

        # Create a dictionary with rounded metrics
        metrics_dict = {
            key.capitalize(): round(value.item() * 100, 2)
            for key, value in test_metrics_values.items()
        }

        # Format the log message
        log_message = (
            f"Loss: {test_loss:.4f} | "
            f"Accuracy: {metrics_dict.get('Accuracy', 'N/A')} | "
            f"Precision: {metrics_dict.get('Precision', 'N/A')} | "
            f"Recall: {metrics_dict.get('Recall', 'N/A')} | "
            f"F1: {metrics_dict.get('F1', 'N/A')} | "
            f"IoU: {metrics_dict.get('Iou', 'N/A')}"
        )

        logger.info(log_message)


if __name__ == "__main__":
    set_seed(42)
    configure_deterministic_behavior()
    device = get_device()
    model = UNet(in_channels=4, out_channels=1).to(device)
    test_metrics = get_metrics()
    test_opt = {
        "batch_size": 64,
        "shuffle": False,
        "num_workers": 8,
        "transform": None,
        "slope": False,
    }

    model_path = Path(
        "/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood/models/concrete_v2/s2_zscore_256_slope_UNet_concrete.pth"
    )

    if "256" in model_path.name:
        test_opt["batch_size"] = 64
    else:
        test_opt["batch_size"] = 16

    if "slope" in model_path.name:
        test_opt["slope"] = True
        if "s1" in model_path.name:
            if "ratio" in model_path.name:
                test_opt["s1_ratio"] = True
                in_channels = 4
            else:
                test_opt["s1_ratio"] = False
                in_channels = 3
        elif "s2" in model_path.name:
            in_channels = 5
        else:
            raise ValueError("Slope model name should contain s1 or s2")
    else:
        test_opt["slope"] = False
        if "s1" in model_path.name:
            if "ratio" in model_path.name:
                test_opt["s1_ratio"] = True
                in_channels = 3
            else:
                test_opt["s1_ratio"] = False
                in_channels = 2
        elif "s2" in model_path.name:
            in_channels = 4
        else:
            raise ValueError("Model name should contain s1 or s2")

    if "UNetPlusPlus" in model_path.name:
        model = UNetPlusPlus(
            in_channels=in_channels, out_channels=1, deep_supervision=False
        ).to(device)
    elif "concrete" in model_path.name:
        print("Using concrete dropout model")
        model = UNetConcrete(in_channels=in_channels, out_channels=1).to(device)
    else:

        model = UNet(in_channels=in_channels, out_channels=1).to(device)
    test(
        model=model,
        test_path=TEST_PATH,
        test_opt=test_opt,
        logger=logger,
        device=device,
        model_path=model_path,
        test_metrics=test_metrics,
        val_or_test=VAL_OR_TEST,
    )
    logger.info("\n")
