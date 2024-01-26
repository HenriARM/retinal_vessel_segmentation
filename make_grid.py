import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_dir = "data/train"


def show(imgs):
    np_imgs = imgs.numpy()
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)))
    plt.savefig("augmentations.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()


data_path = "data"
from dataset import SegmentationDataset

dataset = SegmentationDataset(data_path, "train")
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)


for batch in data_loader:
    images = batch["image"]
    grid = make_grid(images, nrow=4)
    show(grid)
    exit()
