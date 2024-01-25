import os
import numpy as np
from torch.utils.data import Dataset, random_split
import deeplake
import albumentations as A
from skimage.io import imsave
from pathlib import Path


def save_images_and_masks(ds: Dataset, subset_name):
    root = Path("data")
    os.makedirs(root, exist_ok=True)
    subset_dir = root / f"{subset_name}"
    os.makedirs(subset_dir, exist_ok=True)

    augmentation = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.ChannelShuffle(p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.3
            ),
            A.CLAHE(p=0.2),
            A.RandomSnow(p=0.2),
            A.Cutout(p=0.2),
            A.FancyPCA(alpha=0.1, p=0.5),
            A.Solarize(threshold=128, p=0.5),
        ]
    )

    for idx in range(len(ds)):
        image = ds[idx]["rgb_images"].numpy()
        imsave(subset_dir / f"image-idx{idx}_original.png", image)

        if subset_name != "test":
            mask = (255 * ds[idx]["manual_masks/mask"].numpy()[:, :, 0]).astype(np.uint8)
            imsave(subset_dir / f"mask-idx{idx}_original.png", mask)

        if subset_name == "train":
            num_augmentations_per_image = 50
            for i in range(num_augmentations_per_image):
                augmented = augmentation(image=image, mask=mask)
                augmented_image = augmented["image"]
                augmented_mask = augmented["mask"]
                # TODO: save as augm_horizontal_p0.5_vertical_p0.5_rotate90_p0.5.png
                imsave(subset_dir / f"image-idx{idx}-augm{i}.png", augmented_image)
                imsave(subset_dir / f"mask-idx{idx}-augm{i}.png", augmented_mask)


def main():
    # Load datasets
    train_ds = deeplake.load("hub://activeloop/drive-train")
    test_ds = deeplake.load("hub://activeloop/drive-test")

    # Split dataset indices into train, val, and test
    train_size = int(0.8 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [train_size, len(train_ds) - train_size])

    save_images_and_masks(train_ds, "train")
    save_images_and_masks(val_ds, "val")
    save_images_and_masks(test_ds, "test")


if __name__ == "__main__":
    main()

# TODO: from hydra data_path, augmentation configs, e.x. jitter params
# TODO: how to make this faster? (parallelize, numba)
