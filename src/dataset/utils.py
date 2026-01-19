import cv2 as cv
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ProcessPoolExecutor


def img_to_patches(img, patch_size, overlap):
    """
    Splits an image into patches of size patch_size x patch_size with overlap.
    """
    h, w, _ = img.shape
    step = patch_size - overlap

    for i in range(0, h, step):
        for j in range(0, w, step):

            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)
            start_i = end_i - patch_size
            start_j = end_j - patch_size

            yield img[start_i:end_i, start_j:end_j]


def parallel_img_to_patches(patches_dir, dataset_dir, img_path, patch_size, overlap):

    patch_name = patches_dir / img_path.relative_to(dataset_dir)
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image: {img_path}")
        return
    patches = img_to_patches(img, patch_size, overlap)

    for i, img in enumerate(patches):

        if patch_name.suffix == ".jpg":
            img_save_name = patch_name.with_stem(f"{patch_name.stem}_{i+1}")
        elif patch_name.suffix == ".png":
            img_save_name = patch_name.with_stem(f"{patch_name.stem}_{i+1}")
        else:
            raise ValueError("Invalid image extension")

        img_save_name.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(img_save_name, img)


def create_patches(dataset_dir, patch_size=512, overlap=0, patches_dir=None):
    """
    Splits all images in the dataset into patches of size patch_size x patch_size with overlap.
    """

    assert dataset_dir.exists(), f"Path {dataset_dir} does not exist"
    assert (
        dataset_dir.resolve() != patches_dir.resolve()
    ), "Dataset and patches directory cannot be the same"

    if patches_dir is None:
        patches_dir = dataset_dir / "patches"

    if not patches_dir.exists():
        patches_dir.parent.mkdir(parents=True, exist_ok=True)

    imgs_paths = [f for f in dataset_dir.rglob("*") if f.suffix in [".jpg", ".png"]]
    print(f"Found {len(imgs_paths)} images")

    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    parallel_img_to_patches,
                    [patches_dir] * len(imgs_paths),
                    [dataset_dir] * len(imgs_paths),
                    imgs_paths,
                    [patch_size] * len(imgs_paths),
                    [overlap] * len(imgs_paths),
                ),
                total=len(imgs_paths),
            )
        )


def get_transform(is_train=True, mean=None, std=None):
    """
    Returns the appropriate transformation based on whether the user wants training or testing transformations.
    """
    if is_train:
        # Training Transformations
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomCrop(height=128, width=128),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # Testing Transformations
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


if __name__ == "__main__":
    DATASET_DIR = Path("/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood/data")
    PATCHES_DIR = Path(
        "/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood/data/patches"
    )
    create_patches(DATASET_DIR, patch_size=256, overlap=0, patches_dir=PATCHES_DIR)
