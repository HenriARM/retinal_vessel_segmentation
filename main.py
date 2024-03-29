import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
import wandb
import numpy as np


class SegmentationModel(pl.LightningModule):
    def __init__(
        self, arch, encoder_name, in_channels, out_classes, args=None, **kwargs
    ):
        super().__init__()
        # TODO:
        self.save_hyperparameters()
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
        self.loss_dict = {"train": [], "val": [], "test": []}
        self.tp_dict = {"train": [], "val": [], "test": []}
        self.fp_dict = {"train": [], "val": [], "test": []}
        self.fn_dict = {"train": [], "val": [], "test": []}
        self.tn_dict = {"train": [], "val": [], "test": []}
        self.iou_score = {"train": [], "val": [], "test": []}
        self.f1_score = {"train": [], "val": [], "test": []}
        self.f2_score = {"train": [], "val": [], "test": []}
        self.accuracy = {"train": [], "val": [], "test": []}
        self.recall = {"train": [], "val": [], "test": []}

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
        self.loss_dict[stage].append(loss)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        if stage == "val":
            # batch_concatenated_images = []
            for i in range(image.shape[0]):
                img_np = image[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C] format
                mask_np = mask[i].squeeze().cpu().numpy()  # Assuming [1, H, W] format
                pred_mask_np = pred_mask[i].squeeze().cpu().numpy()  # [H, W] format
                # Normalize the image for display if necessary  # TODO:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                wandb.log(
                    {
                        f"{stage}_idx{i}": wandb.Image(
                            img_np,
                            masks={
                                "predictions": {
                                    "mask_data": pred_mask_np,
                                    "class_labels": {0: "vessel_pred"},
                                },
                                "ground_truth": {
                                    "mask_data": mask_np,
                                    "class_labels": {0: "vessel_gt"},
                                },
                            },
                        )
                    }
                )

        # calculate confusion matrix
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        self.tp_dict[stage].append(tp)
        self.fp_dict[stage].append(fp)
        self.fn_dict[stage].append(fn)
        self.tn_dict[stage].append(tn)

        # calculate iou score
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.iou_score[stage].append(iou_score)

        # calculate other metrics
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        self.f1_score[stage].append(f1_score)

        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        self.f2_score[stage].append(f2_score)

        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        self.accuracy[stage].append(accuracy)

        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        self.recall[stage].append(recall)

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
        print(f"Epoch {self.current_epoch} {stage} loss: {self.loss_dict[stage]}")
        # plot loss
        avg_loss = torch.stack(self.loss_dict[stage]).mean()
        self.log(f"loss/{stage}_dice_loss", avg_loss, on_epoch=True, on_step=False)
        self.loss_dict[stage].clear()

        # plot iou score
        iou_score = torch.stack(self.iou_score[stage]).mean()
        self.log(f"metrics/{stage}_iou", iou_score, on_epoch=True, on_step=False)
        self.iou_score[stage].clear()

        # plot f1 score
        f1_score = torch.stack(self.f1_score[stage]).mean()
        self.log(f"metrics/{stage}_f1_score", f1_score, on_epoch=True, on_step=False)
        self.f1_score[stage].clear()

        # plot f2 score
        f2_score = torch.stack(self.f2_score[stage]).mean()
        self.log(f"metrics/{stage}_f2_score", f2_score, on_epoch=True, on_step=False)
        self.f2_score[stage].clear()

        # plot accuracy
        accuracy = torch.stack(self.accuracy[stage]).mean()
        self.log(f"metrics/{stage}_accuracy", accuracy, on_epoch=True, on_step=False)
        self.accuracy[stage].clear()

        # plot recall
        recall = torch.stack(self.recall[stage]).mean()
        self.log(f"metrics/{stage}_recall", recall, on_epoch=True, on_step=False)
        self.recall[stage].clear()

        # TODO: plot confusion matrix
        # tp = torch.stack(self.tp_dict[stage]).mean()
        # fp = torch.stack(self.fp_dict[stage]).mean()
        # fn = torch.stack(self.fn_dict[stage]).mean()
        # tn = torch.stack(self.tn_dict[stage]).mean()

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
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

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
        "TODO"
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

    logger = WandbLogger(project="vessel-segmentation", log_model=True)
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

    trainer = pl.Trainer(
        accelerator="gpu", devices=[0], max_epochs=500, callbacks=[], logger=logger
    )
    trainer.fit(model, train_loader, val_loader)
    # TODO: trainer.test(dataloaders=test_loader)or run on Multiple GPUs so it would be in parallel
