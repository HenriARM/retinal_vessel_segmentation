# retinal_vessel_segmentation

# Data
![example](example.png)

# TODO
* stable diffusion for data augmentation
* Dice loss (for metrics), IoU (Jaccard index)
* U-net squarred
* Focal loss? ask colleagues for tricks in segmentation
* hydra
* interpretebility (gradcam, saliency maps ,captum)
* distributed training
* Mask2Former
* use torch.timm other backbones
* hyperopt

* add something from statistics (confidence tests etc.)


* multi-layer loss


pip install hydra-core



* DiceLoss + FocalLoss https://gitlab.giraffe360-mimosa.com/machine-learning/training/mirror-segmentation-trainer/-/blob/main/scripts/model.py?ref_type=heads