"""Inference module for UNet model."""

import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from model.unet import UNet
from utils.configure_logger import configure_logger
from utils.metrics import compute_dice_score

configure_logger()
logger = logging.getLogger(__name__)

RGB_classes: dict = {
    "Water": [226, 169, 41],
    "Land": [132, 41, 246],
    "Road": [110, 193, 228],
    "Building": [60, 16, 152],
    "Vegetation": [254, 221, 58],
    "Unlabeled": [155, 155, 155],
}

class_to_id: dict = {
    "Water": 1,
    "Land": 2,
    "Road": 3,
    "Building": 4,
    "Vegetation": 5,
    "Unlabeled": 0,
}


def load_model(checkpoint: Path) -> UNet:
    """Load model from path."""
    model: UNet = UNet(n_channels=3, n_classes=6, bilinear=True)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    return model


def mask_to_color_image(mask: np.ndarray) -> np.ndarray:
    """Convert mask to color image."""
    cls_mask = np.zeros(mask.shape + (3,), dtype=np.uint8)
    classes = ["Water", "Land", "Road", "Building", "Vegetation", "Unlabeled"]
    for class_name in classes:
        cls_mask[mask == class_to_id[class_name]] = RGB_classes[class_name]
    return cls_mask


def inference_on_dataloader(
    model: UNet, dataloader: DataLoader, plot: bool = False
) -> None:
    """Inference on given dataloader with ground truth."""
    model.eval()

    for batch_i, (x, y) in enumerate(dataloader):
        for j in range(len(x)):
            with torch.no_grad():
                result = model(x[j : j + 1])

            mask = torch.argmax(result, axis=1)[0]
            gt_mask = y[j]

            dice = compute_dice_score(mask, gt_mask)
            logger.info("Dice score is \n{}".format(dice))

            if plot:
                im = x[j].cpu().detach().numpy()
                im = np.moveaxis(im, 0, -1).copy() * 255

                im = im.astype(int)

                gt_mask = gt_mask.cpu().detach().numpy()
                gt_mask = mask_to_color_image(gt_mask)

                mask = mask.cpu().detach().numpy()
                mask = mask_to_color_image(mask)

                # plot
                plt.figure(figsize=(10, 5), dpi=100)

                plt.subplot(1, 3, 1)
                plt.imshow(im)

                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask)

                plt.subplot(1, 3, 3)
                plt.imshow(mask)

                plt.show()


def inference_on_single_image(
    model: UNet, image_path: Path, plot: bool = True
) -> None:
    """Inference on given dataloader with ground truth."""
    # load the image and preprocess it
    image = Image.open(image_path)
    transform = transforms.transforms.Compose = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )

    input_tensor = transform(image)
    input_tensor = torch.unsqueeze(input_tensor, 0)

    # inference
    with torch.no_grad():
        result = model(input_tensor)
        result = torch.argmax(result, axis=1)[0]

    mask = result.cpu().detach().numpy()
    mask = mask_to_color_image(mask)

    # plot
    if plot:
        plt.figure(figsize=(7, 5), dpi=100)

        plt.subplot(1, 2, 1)
        plt.imshow(image.resize((512, 512)))

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()
