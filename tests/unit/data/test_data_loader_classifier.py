"""Test data loader for classifier."""

from uuid import uuid4

import pytest
from PIL import Image

from data.data_loader_classifier import DataLoader


@pytest.fixture(scope="session")
def image_dir(tmp_path_factory):
    """Fixture to create temporary directory and images."""
    num_classes = 3
    num_images_per_class = 4
    root_dir = tmp_path_factory.mktemp("temp_test_root")
    for _ in range(num_classes):
        class_name = str(uuid4())
        class_dir = root_dir / class_name
        class_dir.mkdir()
        for _ in range(num_images_per_class):
            image_name = class_dir / (str(uuid4()) + ".jpg")
            image = Image.new("RGB", (256, 278))
            image.save(image_name)
    return root_dir


def test_dataloader(image_dir):
    """Test classifier dataloader."""
    data_loader = DataLoader(image_dir)
    sample_image = next(data_loader)[0]
    assert isinstance(sample_image, Image.Image)
    assert sample_image.size == (224, 224)


def test_dataloader_no_images(image_dir):
    """Test classifier dataloader with no images."""
    data_loader = DataLoader(image_dir, image_extension="png")
    with pytest.raises(StopIteration):
        next(data_loader)[0]
