import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import deeplake
from dataset import DRIVECustomDataset
from torchvision import transforms as T


import cv2
import matplotlib.pyplot as plt
import numpy as np

class SegmentationModel(pl.LightningModule):
    def __init__(
        self, arch, encoder_name, in_channels, out_classes, args=None, **kwargs
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        # self.model = smp.create_model(
        #     arch,
        #     encoder_name=encoder_name,
        #     in_channels=in_channels,
        #     classes=out_classes,
        #     **kwargs,
        # )
        self.args = args
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        # image = np.moveaxis(image, -1, 0)
        # print(image)
        # (batch_size, num_channels, height, width)
        assert image.ndim == 4
        # Check that image dimensions are divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"]
        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        self.log(f"loss/{stage} loss", loss)
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        # plot mask predicted if stage is valid
        if stage == "valid":
            predicted_mask = np.moveaxis(pred_mask[0].cpu().numpy(), 0, -1)
            real_mask = np.moveaxis(mask[0].cpu().numpy(), 0, -1)
            # TODO: np.uint8
            image = np.moveaxis(image[0].cpu().numpy(), 0, -1)
            real_mask = np.concatenate([real_mask]*3, axis=-1)*255
            predicted_mask = np.concatenate([predicted_mask]*3, axis=-1)*255

            print(image.shape)
            print(real_mask.shape)
            print(predicted_mask.shape)
            stacked = np.hstack((image, real_mask, predicted_mask))
            cv2.imwrite(f"predicted_mask_{self.current_epoch}.png", stacked)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"per_image_iou/{stage}_per_image_iou": per_image_iou,
            f"dataset_iou/{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

        if stage == "valid":
            hyper_metrics = {
                f"hparam/{stage}_per_image_iou": per_image_iou,
                f"hparam/{stage}_dataset_iou": dataset_iou,
            }
        self.trainer.logger.log_hyperparams(args.__dict__, metrics=hyper_metrics)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    print(parser)
    parser.add_argument("--data_directory", default="./sample_data", type=str)
    parser.add_argument("--img_dirname", default="images", type=str)
    parser.add_argument("--mask_dirname", default="masks", type=str)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--step_lr_step", type=int, default=30)
    parser.add_argument("--step_lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--is_cuda", action="store_true", default=True)
    parser.add_argument("--load_weights", action="store_true", default=False)
    parser.add_argument("--path_to_model", default="./sample_data", type=str)
    parser.add_argument("--dataset_len", default=-1, type=int)
    parser.add_argument("--img_width", default=1024, type=int)
    parser.add_argument("--img_height", default=1024, type=int)
    parser.add_argument("--classes", default=1, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--early_stopping_patience", default=-1, type=int)
    parser.add_argument("--arch", default="Unet", type=str)
    parser.add_argument("--encoder_name", default="resnet34", type=str)
    parser.add_argument("--encoder_weights", default=None, type=str)

    args, other_args = parser.parse_known_args()
    args.arch = "Unet"
    args.encoder_name = "resnet34"
    args.encoder_weights = "imagenet"
    args.data_directory = (
        "/mnt/machine_learning/datasets/sky-segmentation/outside_combined_CAM"
    )
    args.img_dirname = "train_img"
    args.mask_dirname = "train_masks"
    args.num_epochs = 15
    args.dataset_len = 100  # -1
    args.classes = 1
    args.optimizer = "adam"
    args.img_width = 1024
    args.img_height = 768
    args.is_cuda = True
    img_size = (args.img_width, args.img_height)
    
    if not torch.cuda.is_available() or not args.is_cuda:
        args.device = "cpu"
        args.is_cuda = False
        num_workers = os.cpu_count()
        print("cuda not available")
    else:
        args.device = "cuda"
        num_workers = torch.cuda.device_count()
        args.gpu_count = torch.cuda.device_count()
        print(f"cuda devices: {args.gpu_count}")

    metrics = {
        f"per_image_iou/train_per_image_iou": 0,
        f"dataset_iou/train_dataset_iou": 0,
        f"per_image_iou/valid_per_image_iou": 0,
        f"dataset_iou/valid_dataset_iou": 0,
    }

    logger = TensorBoardLogger("tb_logs", default_hp_metric=False)
    hyper_dict = args.__dict__

    logger.log_hyperparams(hyper_dict, metrics=metrics)

    model = SegmentationModel(
        args.arch,
        args.encoder_name,
        in_channels=args.in_channels,
        out_classes=1,
        args=args,
    )
    # TODO
    #   model.save_hyperparameters(hyper_dict.keys())
    #   print(model.hparams)
    #   model.hparams = hyper_dict

    from torch.utils.data import DataLoader, random_split
    transform = T.Compose(
        [
            # TODO: will not work since we need to make sure that flip is applied
            #  to both image and mask at the same time
            # TODO: use albumentation
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # TODO: check picture incorrect_augmentation.png
            # T.RandomRotation(degrees=(0, 360)),
            # only applied to image with mode RGB
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )
    target_transform = T.Compose(
        [
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        ]
    )

    train_dataset = DRIVECustomDataset(
        deeplake.load("hub://activeloop/drive-train"), transform, target_transform
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size= 4, shuffle=False, num_workers=4)

    # TODO: val ds should not have augmentation

    # Model, Trainer, and Training
    # model = UNet()
    # val_images, val_masks = next(iter(val_loader))
    # image_prediction_logger = ImagePredictionLogger(val_samples=(val_images, val_masks))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=500,
        callbacks=[],
        # logger=logger
    )
    trainer.fit(model, train_loader, val_loader)

