# retinal_vessel_segmentation

# Data
![example](resources/example.png)

# Augmentation
![augmentation](resources/augmentations.png)

# Training (fucked color channel)
![0](resources/predicted_mask_0.png)
![1](resources/predicted_mask_1.png)
![2](resources/predicted_mask_2.png)
![3](resources/predicted_mask_3.png)

# Test example
![114](resources/predicted_mask_114.png)
![217](resources/predicted_mask_217.png)


# Installation
```bash
$ pip install -r requirements.txt
$ python prepare_dataset.py
$ python main.py
```

# TODO
* add checkpoint backbone
* suppersampling or upsampling back resolution (GuidedConvFilter add to the architecture)
* stable diffusion for data augmentation
* U-net squarred
* hydra
* interpretebility (gradcam, saliency maps ,captum)
* distributed training
* Mask2Former
* use torch.timm other backbones
* hyperopt
* multi-layer loss
* DiceLoss + FocalLoss https://gitlab.giraffe360-mimosa.com/machine-learning/training/mirror-segmentation-trainer/-/blob/main/scripts/model.py?ref_type=heads


pip install hydra-core

You are using a CUDA device ('NVIDIA RTX A4000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision