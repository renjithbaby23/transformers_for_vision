"""Test data loader for segmentation."""

from pathlib import Path

from data.data_loader_segmentation import (
    InfiniteSegDataLoader,
    SegmentationDataset,
)


def test_dataset():
    """Test segmentation dataloader."""
    image_dir = Path(
        "../../../artefacts/satellite_image_segmentation_dataset/val"
    )
    assert image_dir.is_dir(), "Test data directory doesn't exist!"
    dataset = SegmentationDataset(image_dir)
    assert len(dataset) > 0
    sample_image, sample_label = dataset[0]
    assert sample_image.shape == (3,) + dataset.image_size
    assert sample_label.shape == dataset.image_size


def test_dataset_no_images():
    """Test segmentation dataset with no images."""
    image_dir = Path("../../../artefacts/satellite_image_segmentation_dataset")
    assert image_dir.is_dir(), "Test data directory doesn't exist!"

    dataset = SegmentationDataset(image_dir)
    assert len(dataset) == 0


def test_dataloader_infinite():
    """Test segmentation infinite dataloader."""
    image_dir = Path(
        "../../../artefacts/satellite_image_segmentation_dataset/val"
    )
    assert image_dir.is_dir(), "Test data directory doesn't exist!"
    batch_size = 3
    data_loader = InfiniteSegDataLoader(
        image_dir, train=True, shuffle=True, batch_size=batch_size
    )

    assert len(data_loader) > 0

    for i in range(len(data_loader) // batch_size * 2):
        next(data_loader)

    sample_image, sample_label = next(data_loader)
    assert (
        sample_image.shape == (batch_size, 3) + data_loader.dataset.image_size
    )
    assert sample_label.shape == (batch_size,) + data_loader.dataset.image_size
