import os
import torch
import wandb
from torchvision.transforms import functional as TF
from segmentation_models_pytorch import Unet
from PIL import Image


project_name = "sky-segmentation"
entity = "henrikgabrielyan"
#TODO: why init? 
# wandb.init(project=project_name, entity=entity)
model = Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
)
# run_path = f"{entity}/{project_name}/vz0r59ge"
# artifact = wandb.use_artifact(f"{run_path}/model-best:v0", type="model")
# artifact_dir = artifact.download()
# checkpoint_path = os.path.join(artifact_dir, "model-best.pt")

checkpoint_path = "/home/henri/Desktop/retinal_vessel_segmentation/sky-segmentation/vz0r59ge/checkpoints/epoch=71-step=14688.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
image = Image.open("/home/henri/Desktop/retinal_vessel_segmentation/data/test/images/image-idx14_original.png")
input_tensor = TF.to_tensor(image).unsqueeze(0)
with torch.no_grad():
    prediction = model(input_tensor)
# Post-processing if necessary, e.g., apply threshold to get binary mask
predicted_mask = prediction.squeeze().numpy()
binary_mask = predicted_mask > 0.5  # Example thresholding step
print(binary_mask.shape)
# Do something with the predicted mask, like saving or displaying
