# retinal_vessel_segmentation

# Data
![example](example.png)

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