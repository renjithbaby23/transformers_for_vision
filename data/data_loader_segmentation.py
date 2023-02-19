"""Finite dataloader for for segmentation data."""

from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SegmentationDataset(Dataset):
    """Segmentation dataset."""

    def __init__(
        self,
        root_dir: Path,
        train: bool = True,
        transform: Optional[transforms.transforms.Compose] = None,
    ):
        """Segmentation dataset.

        Args:
            root_dir: segmentation dataset root directory
            train: train dataset or not
            transform: transforms to apply on input data
        """
        super().__init__()
        self.root_dir: Path = root_dir
        self.transform: transforms.transforms.Compose = transform
        self.image_size: tuple[int, int] = (512, 512)
        self.train: bool = train
        self.IMG_NAMES: list[Path] = sorted(root_dir.glob("*/images/*.jpg"))

        self.RGB_classes: dict = {
            "Water": [226, 169, 41],
            "Land": [132, 41, 246],
            "Road": [110, 193, 228],
            "Building": [60, 16, 152],
            "Vegetation": [254, 221, 58],
            "Unlabeled": [155, 155, 155],
        }  # in RGB

        self.class_to_id: dict = {
            "Water": 1,
            "Land": 2,
            "Road": 3,
            "Building": 4,
            "Vegetation": 5,
            "Unlabeled": 0,
        }

        self.id_to_class: dict = {v: k for k, v in self.class_to_id.items()}

        self.pre_process: transforms.transforms.Compose = transforms.Compose(
            [transforms.Resize(self.image_size), transforms.ToTensor()]
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item with index.

        Args:
            idx: index of item to retrieve from the dataset.

        Returns: tuple of tensors for image and mask.
        """
        img_path = self.IMG_NAMES[idx]
        mask_path = Path(
            str(img_path).replace("images", "masks").replace(".jpg", ".png")
        )

        image: Image.Image = Image.open(img_path)
        mask: Image.Image = Image.open(mask_path).convert("RGB")
        resize: Callable = transforms.Resize(
            size=self.image_size,
            interpolation=transforms.functional.InterpolationMode.NEAREST,
        )
        mask = resize(mask)

        mask = np.array(mask)
        cls_mask = np.zeros(mask.shape)

        for class_ in self.class_to_id.keys():
            cls_mask[mask == self.RGB_classes[class_]] = self.class_to_id[
                class_
            ]

        cls_mask = cls_mask[:, :, 0]

        if self.train and self.transform is not None:
            image = self.transform(image)
            image = np.array(image)

            # random vertical flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 0)
                cls_mask = np.flip(cls_mask, 0)

            # random horizontal flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 1)
                cls_mask = np.flip(cls_mask, 1)
            image = transforms.functional.to_pil_image(image)

        image_tensor = self.pre_process(image)

        return image_tensor, torch.tensor(cls_mask.copy(), dtype=torch.int64)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.IMG_NAMES)


class InfiniteSegDataLoader(object):
    """Infinite segmentation data loader class."""

    def __init__(
        self,
        root_dir: Path,
        train: bool = True,
        batch_size: int = 2,
        shuffle: bool = True,
        transform: Optional[transforms.transforms.Compose] = None,
    ):
        """Infinite segmentation data loader.

        Args:
            root_dir: segmentation dataset root directory
            train: train data loader or not
            batch_size: batch size
            shuffle: shuffle or not
            transform: optional transformations on the data
        """
        self.root_dir: Path = root_dir
        self.train: bool = train
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.transform = transform

        self.dataset: Optional[Dataset] = None
        self.dataloader: Optional[DataLoader] = None
        self.iterable: Iterator

        self.reset_data()

    def reset_data(self) -> None:
        """Reset dataloader."""
        self.dataset = SegmentationDataset(
            self.root_dir, train=self.train, transform=self.transform
        )
        # Note that a batch size of 1 must be used here.
        # The actual batch size will be used in the __next__
        self.dataloader = DataLoader(
            self.dataset, batch_size=1, shuffle=self.shuffle
        )
        self.iterable = iter(self.dataloader)

    def __iter__(self):
        """Iterator."""
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Next.

        Enables infinite call for next by catching the StopIteration exception
        and then starting the iteration again.
        """
        images = torch.tensor([])
        labels = torch.tensor([])
        try:
            for count in range(self.batch_size):
                a, b = next(self.iterable)
                images = torch.cat([images, a], dim=0)
                labels = torch.cat([labels, b], dim=0)
        except StopIteration:
            self.reset_data()
            # resume counting form where the iteration stopped
            for _ in range(count, self.batch_size):
                a, b = next(self.iterable)
                images = torch.cat([images, a], dim=0)
                labels = torch.cat([labels, b], dim=0)
        return images, labels

    def __len__(self) -> int:
        """Get actual length of the data loader."""
        return len(self.dataloader)


def display_images_from_dataloader(
    dataset: SegmentationDataset, idx: int = 0
) -> None:
    """Helper to display an image and its mask from dataloader."""
    d = dataset[idx]

    # image
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    # converting from batch first to sequence first
    plt.imshow(np.moveaxis(d[0].numpy(), 0, -1))

    # mask
    plt.subplot(1, 2, 2)
    plt.imshow(d[1].numpy())

    plt.show()


if __name__ == "__main__":
    train_dir = Path("../artefacts/satellite_image_segmentation_dataset/val")

    color_shift = transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3
    )
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])

    train_dataset = SegmentationDataset(train_dir, transform=t)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    print(f"{len(train_dataloader) = }")
    img, label = next(iter(train_dataloader))
    print(f"{img.shape = }")
    print(f"{label.shape = }")

    display_images_from_dataloader(train_dataset)

    # infinite dataloader
    batch_size = 3
    idl = InfiniteSegDataLoader(
        train_dir, train=True, shuffle=True, batch_size=batch_size, transform=t
    )

    print(f"{len(idl) = }")
    for i in range(len(idl) // batch_size * 2):
        img, label = next(idl)
        print(f"{i = }")
        print(f"{img.shape = }")
        print(f"{label.shape = }")
        print("--" * 30)
