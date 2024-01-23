import deeplake
import torch

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.detection import IntersectionOverUnion
from PIL import Image


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
        self.iou_metric = IntersectionOverUnion()

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
        outputs = self(images)
        # TODO: tmp
        outputs = torch.squeeze(outputs, axis=1)
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        outputs = torch.squeeze(outputs, axis=1)
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
        # # Calculate IoU (assuming binary segmentation)
        # preds = torch.sigmoid(outputs) > 0.5  # Convert to binary predictions
        # self.iou_metric.update(preds, targets)
        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # def on_validation_epoch_end(self, outputs):
    #     # Compute and log IoU at the end of the epoch
    #     iou_score = self.iou_metric.compute()
    #     self.log("val_iou", iou_score, on_epoch=True, prog_bar=True)
    #     # Reset IoU metric for the next epoch
    #     self.iou_metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DRIVECustomDataset(Dataset):
    def __init__(self, deeplake_dataset, transform=None, target_transform=None):
        """
        Custom dataset for the DRIVE dataset loaded via DeepLake.
        Args:
            deeplake_dataset: A DeepLake dataset object.
            transform: Optional transform to be applied on a sample.
        """
        self.deeplake_dataset = deeplake_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.deeplake_dataset["rgb_images"])

    def __getitem__(self, idx):
        # Load the images and masks
        image = Image.fromarray(self.deeplake_dataset["rgb_images"][idx].numpy())
        if self.transform:
            image = self.transform(image)

        target = self.deeplake_dataset["manual_masks/mask"][idx].numpy()[:, :, 0]
        target = Image.fromarray(target)
        if self.target_transform:
            target = self.target_transform(target)
        target = np.array(target)
        target[target != 0] = 1
        target = torch.from_numpy(target).long()
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

    train_dataset = DRIVECustomDataset(
        deeplake.load("hub://activeloop/drive-train"), transform, target_transform
    )
    # TODO:
    # test_dataset = DRIVECustomDataset(deeplake.load("hub://activeloop/drive-test"))
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

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
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

# TODO: add checkpoint loading/saving
# TODO: add augmentation
# TODO: suppersampling or upsampling back resolution
