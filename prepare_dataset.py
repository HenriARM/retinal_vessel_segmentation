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
        ]
    )

    for idx in range(len(ds)):
        image = ds[idx]["rgb_images"].numpy()
        # TODO: depends on test/train
        mask = (255 * ds[idx]["manual_masks/mask"].numpy()[:, :, 0]).astype(np.uint8)

        if subset_name == "train":
            num_augmentations_per_image = 10
            for i in range(num_augmentations_per_image):
                augmented = augmentation(image=image, mask=mask)
                augmented_image = augmented["image"]
                augmented_mask = augmented["mask"]
                imsave(subset_dir / f"image-idx{idx}-augm{i}.png", augmented_image)
                imsave(subset_dir / f"mask-idx{idx}-augm{i}.png", augmented_mask)
        else:
            imsave(subset_dir / f"image-idx{idx}.png", image)
            imsave(subset_dir / f"mask-idx{idx}.png", mask)


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
