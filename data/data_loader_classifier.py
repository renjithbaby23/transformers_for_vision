"""Infinite data loader for image classification tasks."""
import random
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Iterable, Literal, Union

from PIL import Image
from sklearn.preprocessing import LabelEncoder

image_type = Literal["jpg", "png"]


class DataLoader(object):
    """Image dataloader class sample."""

    def __init__(
        self,
        image_dir: str,
        batch_size: int = 4,
        image_extension: image_type = "jpg",
        image_dim: tuple[int, int] = (224, 224),
        shuffle: bool = True,
        infinite: bool = True,
        return_labels: bool = False,
        augment: bool = False,
    ):
        """Dataloader class to load images from a given directory.

        This implements an infinite data loader.

        Args:
            image_dir: directory containing class directories and images in class directories
            batch_size: data loader batch size
            image_extension: image file extension
            image_dim: image dimension to which each input image will be resized
            shuffle: should shuffle the image order or not
            infinite: should it be an infinite dataloader or finite one
            return_labels: should the dataloader return labels as well
            augment: should use augmentations or not

        Note: This dataloader returns a list of PIL images, not tensors.
        """
        self.image_dir: Path = Path(image_dir).resolve()
        self.root_dir: str = image_dir
        self.batch_size: int = batch_size
        self.image_extension: image_type = image_extension
        self.image_dim: tuple[int, int] = image_dim
        self.shuffle: bool = shuffle
        self.infinite: bool = infinite
        self.return_labels: bool = return_labels
        self.augment: bool = augment

        self.images: list[Path] = list()
        self.pool: Iterable[Path]
        self.n_images: int = 0

        self.classes: list = list()
        self.n_classes: int = 0
        self.class_stats: dict = dict()
        self.class_to_id: dict = dict()
        self.id_to_class: dict = dict()
        self._update()

    def _update(self, random_seed: int = 42):
        """Update dataloader with details from the image directory."""
        self.images = list(self.image_dir.glob(f"*/*.{self.image_extension}"))
        if self.shuffle:
            random.seed(random_seed)
            random.shuffle(self.images)
        self.n_images = len(self.images)

        self.class_stats = dict(Counter([img.parent.name for img in self.images]))
        self.classes = list(self.class_stats.keys())
        self.n_classes = len(self.class_stats)

        # create an infinite generator from the input images list
        if self.infinite:
            self.pool = cycle(self.images)
        else:
            self.pool = iter(self.images)

        le = LabelEncoder()
        le.fit(self.classes)
        self.class_to_id = {item: le.transform([item])[0] for item in self.classes}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

    def __iter__(self):
        """Iterator."""
        return self

    def __next__(self):
        """Next implementation of the iterable.

        Note: modify here if tensor needs to be returned.
        """
        if self.return_labels:
            images = []
            labels = []
            for _ in range(self.batch_size):
                path = next(self.pool)
                images.append(self.load_images(path))
                labels.append(self.class_to_id[path.parent.name])
            return images, labels
        else:
            images = [self.load_images(next(self.pool)) for _ in range(self.batch_size)]
            return images

    def load_images(self, image_path: Path) -> Image.Image:
        """Open and load image from a given path."""
        image = Image.open(image_path).resize(self.image_dim)
        if self.augment:
            image = self.augmentation_func(image)
        return image

    @staticmethod
    def augmentation_func(image):
        """Image augmentations."""
        # Implement complex augmentations as needed.
        # Using identity transform now.
        return image


if __name__ == "__main__":
    dl = DataLoader(
        "../artefacts/satellite_image_classification_dataset",
        5,
        infinite=True,
        return_labels=True,
    )

    for i in range(5):
        x = next(dl)
        # print(x[0][0].show())
        print(x[1][0])
        print(f"{len(x[0]) = }")
