import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
import wandb


class SegmentationModel(pl.LightningModule):
    def __init__(
        self, arch, encoder_name, in_channels, out_classes, args=None, **kwargs
    ):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.args = args
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # TODO:
        self.training_step_outputs = {
            "tp": [],
            "fp": [],
            "fn": [],
            "tn": [],
        }
        self.validation_step_outputs = {
            "tp": [],
            "fp": [],
            "fn": [],
            "tn": [],
        }

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, batch_idx, stage):
        image = batch["image"]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        self.log(f"loss/{stage} loss", loss)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        # TODO:
        step_outputs = (
            self.training_step_outputs
            if stage == "train"
            else self.validation_step_outputs
        )
        step_outputs["tp"].append(tp)
        step_outputs["fp"].append(fp)
        step_outputs["fn"].append(fn)
        step_outputs["tn"].append(tn)

        # self.add_images_to_tensorboard(batch_idx, image, mask, pred_mask, stage)

        return loss

    def reconstruct_labels(self, tp, fp, fn, tn):
        y_true = []
        y_pred = []

        # True Positives: Both predicted and true labels are Positive
        y_true.extend(["Positive"] * tp)
        y_pred.extend(["Positive"] * tp)

        # False Positives: Predicted Positive but actually Negative
        y_true.extend(["Negative"] * fp)
        y_pred.extend(["Positive"] * fp)

        # False Negatives: Predicted Negative but actually Positive
        y_true.extend(["Positive"] * fn)
        y_pred.extend(["Negative"] * fn)

        # True Negatives: Both predicted and true labels are Negative
        y_true.extend(["Negative"] * tn)
        y_pred.extend(["Negative"] * tn)

        return y_true, y_pred

    def shared_epoch_end(self, stage):
        step_outputs = (
            self.training_step_outputs
            if stage == "train"
            else self.validation_step_outputs
        )

        tp = torch.stack(step_outputs["tp"])
        fp = torch.stack(step_outputs["fp"])
        fn = torch.stack(step_outputs["fn"])
        tn = torch.stack(step_outputs["tn"])
        # TODO: why tp.shape is [204,4,1]
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # TODO: shouldn't both be same?
        self.log(f"{stage}_per_image_iou", per_image_iou, on_epoch=True)
        self.log(f"{stage}_dataset_iou", dataset_iou, on_epoch=True)

        print("Done")
        for key in step_outputs.keys():
            step_outputs[key].clear()

        # y_true, y_pred = self.reconstruct_labels(tp, fp, fn, tn)
        # wandb.log(
        #     {
        #         f"{stage}_confusion_matrix": wandb.plot.confusion_matrix(
        #             probs=None, y_true=y_true, y_pred=y_pred, class_names=["vessel"]
        #         ),
        #         "epoch": self.current_epoch,
        #     }
        # )

    def training_step(self, batch, batch_idx):
        self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        self.shared_epoch_end("valid")

    def on_test_epoch_end(self):
        self.shared_epoch_end("test")

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

    # Initialize wandb
    wandb.init(project="vessel_segmentation", entity="henrikgabrielyan")

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

    logger = WandbLogger(project="sky-segmentation", log_model=True)
    hyper_dict = args.__dict__
    logger.log_hyperparams(hyper_dict)

    model = SegmentationModel(
        args.arch,
        args.encoder_name,
        in_channels=args.in_channels,
        out_classes=1,
        args=args,  # TODO:
    )
    # TODO
    #   model.save_hyperparameters(hyper_dict.keys())
    #   print(model.hparams)
    #   model.hparams = hyper_dict

    data_path = "data"
    train_dataset = SegmentationDataset(data_path, "train")
    val_dataset = SegmentationDataset(data_path, "val")
    test_dataset = SegmentationDataset(data_path, "test")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # TODO:
    # val_images, val_masks = next(iter(val_loader))
    # image_prediction_logger = ImagePredictionLogger(val_samples=(val_images, val_masks))

    trainer = pl.Trainer(
        accelerator="gpu", devices=[0], max_epochs=500, callbacks=[], logger=logger
    )
    # TODO: test loader (write separately shared step)
    trainer.fit(model, train_loader, val_loader)


"""
val_images, val_masks = next(iter(val_loader))
image_prediction_logger = ImagePredictionLogger(val_samples=(val_images, val_masks))

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
"""
