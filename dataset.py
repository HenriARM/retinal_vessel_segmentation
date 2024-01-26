import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, data_path, subset, transform=None, mask_transform=None):
        """
        Custom dataset for image segmentation.
        Args:
            data_path (Path or str): Path to the dataset directory.
            subset (str): Sub-directory for the dataset ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on the image.
            target_transform (callable, optional): Optional transform to be applied on the mask.
        """
        self.data_path = Path(data_path) / subset
        self.images_dir = self.data_path / "images"
        self.masks_dir = self.data_path / "masks"
        self.images = sorted(list(self.images_dir.glob("*.png")))
        self.masks = sorted(list(self.masks_dir.glob("*.png")))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_size = 256, 256  # original size is 584, 565

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.img_size)
        mask = mask.resize(self.img_size)
        image = np.asarray(image)
        mask = np.asarray(mask)
        mask = mask / 255.0  # np.unique(mask.flatten())
        mask = np.where(mask > 0.5, 1, 0)
        sample = {}
        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        return sample
