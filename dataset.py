from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

import torchvision.transforms as transforms


class DRIVECustomDataset(Dataset):
    def __init__(self, deeplake_dataset, transform=None, target_transform=None):
        """
        Custom dataset for the DRIVE dataset loaded via DeepLake.
        Args:
            deeplake_dataset: A DeepLake dataset object.
            transform: Optional transform to be applied on a sample.
        """
        self.deeplake_dataset = deeplake_dataset
        # self.transform = transform

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_size = 256, 256
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.deeplake_dataset["rgb_images"])

    def __getitem__(self, idx):
        # Load the images and masks
        image = Image.fromarray(self.deeplake_dataset["rgb_images"][idx].numpy())
        image = image.convert("RGB")
        # if self.transform:
        #     image = self.transform(image)

        target = Image.fromarray(
            self.deeplake_dataset["manual_masks/mask"][idx].numpy()[:, :, 0]
        )
        target = target.convert("L")

        image_transform = image.resize(self.img_size)
        mask_transform = target.resize(self.img_size)
        img = np.asarray(image_transform)
        mask = np.asarray(mask_transform)
        mask = mask / 255.0

        img_tensor = self.transform(img)
        mask_tensor = self.transform(mask)

        y_other = torch.ones(
            [1, mask_tensor.shape[1], mask_tensor.shape[2]]
        ) - torch.sum(mask_tensor, dim=0)
        mask_tensor = torch.cat((mask_tensor, y_other), dim=0)
        # plt.imshow(mask_tensor[0], cmap="gray")

        # TODO:
        sample = dict(image=img_tensor, mask=mask, trimap=mask_tensor)
        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(img, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        return sample

        # if self.target_transform:
        #     target = self.target_transform(target)
        # target = np.array(target)
        # target[target != 0] = 1
        # target = torch.from_numpy(target).long()
        # return image, target


# TODO: convert mask

    # a = image.to("cpu")[0].numpy()
    # a=np.moveaxis(a, 0, -1)
    # plt.imshow(a, cmap="gray")