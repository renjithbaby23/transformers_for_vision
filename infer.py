"""Model inference entrypoint."""

from pathlib import Path

from torch.utils.data import DataLoader

from data.data_loader_segmentation import SegmentationDataset
from model import inference


def predict_data_loader(test_dir: Path):
    """Predict from a directory."""
    test_dataset = SegmentationDataset(test_dir)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=0
    )

    inference.inference_on_dataloader(model, test_dataloader, plot=True)


def predict_one_image(image_path: Path):
    """Predict on a single image."""
    inference.inference_on_single_image(model, image_path, plot=True)


if __name__ == "__main__":
    checkpoint_path = Path("./artefacts/checkpoint/best_model.pth")
    model = inference.load_model(checkpoint_path)

    infer_on_val_data = False
    infer_on_single_image = True

    if infer_on_val_data:
        sample_dir = Path(
            "./artefacts/satellite_image_segmentation_dataset/val"
        )
        predict_data_loader(sample_dir)
    if infer_on_single_image:
        sample_image = Path(
            "./artefacts/satellite_image_segmentation_dataset/"
            "val/Tile 1/images/image_part_009.jpg"
        )
        predict_one_image(sample_image)
