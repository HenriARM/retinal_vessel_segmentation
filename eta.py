# pip install deeplake

import cv2
import numpy as np
import deeplake

# Load the datasets
# https://datasets.activeloop.ai/docs/ml/datasets/drive-dataset/
train_ds = deeplake.load("hub://activeloop/drive-train")
# test_ds = deeplake.load("hub://activeloop/drive-test")
# train_dataloader = train_ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
# test_dataloader = test_ds.pytorch(num_workers=0, batch_size=4, shuffle=False)


# Loop through the dataset
for i in range(len(train_ds["rgb_images"])):
    # Load the images and convert them to the proper format
    image = train_ds["rgb_images"][i].numpy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    manual_mask = 255 - (train_ds["manual_masks/mask"][i].numpy() * 255)[
        :, :, 0
    ].astype(np.uint8)
    mask = 255 - (train_ds["masks/mask"][i].numpy() * 255)[:, :, 0].astype(np.uint8)

    manual_mask_3c = cv2.cvtColor(manual_mask, cv2.COLOR_GRAY2BGR)
    mask_3c = cv2.cvtColor(mask[:, :, np.newaxis], cv2.COLOR_GRAY2BGR).astype(np.uint8)

    concatenated_image = np.hstack((image, manual_mask_3c, mask_3c))

    cv2.imshow("Image", concatenated_image)
    key = cv2.waitKey(0)

    # Close the window when 'q' key is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()


# TODO: why we need circular mask? we can send to NN to use it also
# TODO: what's one the second dimension of masks?
# TODO: which format save all pics for faster use? .numpy, .safetensors by HF
