"""Train model."""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from data.data_loader_segmentation import SegmentationDataset
from model.callbacks import EarlyStopping, LRScheduler, SaveBestModel
from model.loss import FocalLoss
from model.unet import UNet
from utils.configure_logger import configure_logger
from utils.metrics import compute_accuracy

configure_logger()
logger = logging.getLogger(__name__)


def get_transforms() -> transforms.Compose:
    """Add transforms and return it.

    Note: Only color transformations are supported now.
    """
    color_shift = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2
    )
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    sharpened = transforms.RandomAdjustSharpness(0.2, 0.4)
    contrast = transforms.RandomAutocontrast(0.4)

    return transforms.Compose([color_shift, blurriness, sharpened, contrast])


def get_dataloaders(
    path: Path,
    batch_size: int = 2,
    shuffle: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:
    """Get the train and test dataloaders."""
    dataset = SegmentationDataset(path, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    return dataloader


class Trainer(object):
    """Model training class."""

    def __init__(self, config: DictConfig):
        """Abstraction for model training."""
        self.config: DictConfig = config
        self.save_dir: Path = Path(config.data.checkpoint)
        os.makedirs(self.save_dir, exist_ok=True)

        self.model: UNet
        self.criterion: FocalLoss
        self.optimizer: torch.optim.optimizer.Optimizer
        self.lr_scheduler: Optional[LRScheduler] = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.save_best: Optional[SaveBestModel] = None

        self.train_dataloader: DataLoader
        self.test_dataloader: DataLoader

        self.loss_list: list = []
        self.acc_list: list = []
        self.val_loss_list: list = []
        self.val_acc_list: list = []

        self.epoch: int = 0
        self.n_epochs: int = config.n_epochs

        self._build()
        self._set_callbacks()
        self._set_dataloaders()

    def _build(self):
        """Build model."""
        self.model = UNet(
            n_channels=self.config.unet.n_channels,
            n_classes=self.config.unet.n_classes,
            bilinear=self.config.unet.bilinear,
        )
        if self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.lr
            )
        else:
            raise ValueError("Only supports 'adam' optimiser!")
        self.criterion = FocalLoss(
            gamma=self.config.loss.gamma, alpha=self.config.loss.alpha
        )

    def _set_callbacks(self):
        """Setup callbacks."""
        if self.config.lr_scheduler.enable:
            self.lr_scheduler = LRScheduler(
                self.optimizer,
                patience=self.config.lr_scheduler.patience,
                min_lr=self.config.lr_scheduler.min_lr,
                factor=self.config.lr_scheduler.factor,
            )
        if self.config.early_stopping.enable:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping.patience,
                min_delta=self.config.early_stopping.min_delta,
            )
        self.save_best = SaveBestModel(self.save_dir)

    def _set_dataloaders(self):
        """Get dataloaders."""
        self.train_dataloader = get_dataloaders(
            Path(self.config.data.train_dir),
            shuffle=True,
            transform=get_transforms(),
        )
        self.test_dataloader = get_dataloaders(
            Path(self.config.data.test_dir), shuffle=False, transform=None
        )

        # load previous checkpoint
        if self.config.resume is not None:
            checkpoint_path = Path(self.config.resume)
            logger.info(f"Resuming from checkpoint - {checkpoint_path}")
            try:
                self.model.load_state_dict(torch.load(checkpoint_path))
            except FileNotFoundError as e:
                logger.warning(e)
                logger.warning("starting training without using checkpoint.")

    def _reset_fit(self):
        """Reset training metrics."""
        self.loss_list = []
        self.acc_list = []

    def _reset_eval(self):
        """Reset validation metrics."""
        self.val_loss_list = []
        self.val_acc_list = []

    def fit(self) -> float:
        """Train loop."""
        self.model.train()
        self._reset_fit()
        for batch_i, (x, y) in enumerate(self.train_dataloader):
            pred_mask = self.model(x)  # [4,6,512,512]
            loss = self.criterion(pred_mask, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_list.append(loss.cpu().detach().numpy())
            self.acc_list.append(compute_accuracy(y, pred_mask).numpy())

            logger.info(
                f"[Epoch {self.epoch}/{self.n_epochs}]"
                f" [Batch {batch_i}/{len(self.train_dataloader)}] "
                f"[Loss: {loss.cpu().detach().numpy()} "
                f"({np.mean(self.loss_list)})]"
            )
        return np.mean(self.loss_list)

    def eval(self) -> float:
        """Validation loop."""
        self.model.eval()
        self._reset_eval()
        for batch_i, (x, y) in enumerate(self.test_dataloader):
            with torch.no_grad():
                pred_mask = self.model(x)
            val_loss = self.criterion(pred_mask, y)
            self.val_loss_list.append(val_loss.cpu().detach().numpy())
            self.val_acc_list.append(compute_accuracy(y, pred_mask).numpy())
        return np.mean(self.val_loss_list)

    def train(self) -> list[tuple[int, float, float]]:
        """Training epochs."""
        epoch_losses = []

        for epoch in range(self.n_epochs):
            self.epoch += 1
            train_loss = self.fit()
            val_loss = self.eval()
            epoch_losses.append((epoch, val_loss, train_loss))
            self._log_epoch_summary()

            if self.lr_scheduler is not None:
                self.lr_scheduler(val_loss)
            if self.early_stopping is not None:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    break
            self.save_best(self.model, self.epoch, val_loss)

        return epoch_losses

    def _log_epoch_summary(self):
        """Log epoch summary."""
        logger.info(
            "\nepoch {} - loss : {:.5f} - acc : {:.4f}"
            "val loss : {:.5f} - val acc : {:.4f}".format(
                self.epoch,
                np.mean(self.loss_list),
                np.mean(self.acc_list),
                np.mean(self.val_loss_list),
                np.mean(self.val_acc_list),
            )
        )


def plot_loss(losses: list[tuple[int, float, float]]) -> None:
    """Plot training loss."""
    loss: np.ndarray = np.array(losses)
    plt.plot(loss[:, 0], loss[:, 1], color="b", linewidth=4)
    plt.plot(loss[:, 0], loss[:, 2], color="r", linewidth=4)
    plt.title("FocalLoss", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.grid()
    plt.legend(["training", "validation"])
    save_path = "./artefacts/checkpoint/loss.png"
    logger.info(f"Saving the loss plot in {save_path}")
    plt.savefig(save_path)
