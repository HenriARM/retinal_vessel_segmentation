import torch

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import VOCSegmentation
import numpy as np
import matplotlib.pyplot as plt


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=3):
        super().__init__()
        self.val_samples = val_samples
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 10 != 0:
            return
        pl_module.eval()
        images, masks = self.val_samples
        with torch.no_grad():
            preds = pl_module(images.to(pl_module.device))
        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()

        for i in range(self.num_samples):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(
                images[i].permute(1, 2, 0)
            )  # Assuming images are in [C, H, W] format
            ax[0].set_title("Input Image")
            ax[1].imshow(masks[i].squeeze())  # Assuming masks are in [1, H, W] format
            ax[1].set_title("Ground Truth Mask")
            ax[2].imshow(preds[i].squeeze())  # Assuming preds are in [1, H, W] format
            ax[2].set_title("Predicted Mask")
            plt.savefig(f"output_{i}_{trainer.current_epoch}.png")
        pl_module.train()


class UNet(pl.LightningModule):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(64, 1, kernel_size=1),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # Decoder
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        print("Image:", images.shape)
        outputs = self(images)
        print("Output:", outputs.shape)
        print("Mask:", targets.shape)
        # tmo
        outputs = torch.squeeze(outputs, axis=1)
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class CustomVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root,
        year="2012",
        image_set="train",
        download=False,
        transform=None,
        target_transform=None,
    ):
        super(CustomVOCDataset, self).__init__(
            root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index):
        image, target = super(CustomVOCDataset, self).__getitem__(index)
        target = np.array(target)
        # Convert to binary mask (foreground/background)
        target[target != 0] = 1
        target = torch.from_numpy(target).long()
        # TODO: why long? int64?
        return image, target


def main():
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    target_transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.NEAREST
            )
        ]
    )

    # Dataset and Data Loader
    train_dataset = CustomVOCDataset(
        root="data",
        year="2012",
        image_set="train",
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = CustomVOCDataset(
        root="data",
        year="2012",
        image_set="val",
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, Trainer, and Training
    model = UNet()

    val_images, val_masks = next(iter(val_loader))
    image_prediction_logger = ImagePredictionLogger(val_samples=(val_images, val_masks))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=100,
        callbacks=[image_prediction_logger],
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
