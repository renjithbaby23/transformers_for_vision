"""Callbacks classes."""
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__file__)


class EarlyStopping(object):
    """Early stopping class.

    To stop the training when the loss does not improve after certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """Early stopping init."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """Check if it is time to early stop."""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            logger.info(
                f"Early stopping counter {self.counter} of {self.patience}"
            )
            if self.counter >= self.patience:
                logger.info("Early stopping...")
                self.early_stop = True


class LRScheduler(object):
    """Learning rate scheduler.

    If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """Learning rate scheduler init."""
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        """Check and update learning rate if needed."""
        logger.info(f"Calling lr scheduler with val_loss {val_loss:.3f}")
        logger.info(f"current lr: {self.optimizer.param_groups[0]['lr']}")
        self.lr_scheduler.step(val_loss)


class SaveBestModel(object):
    """Class to save the best model while training.

    If the current epoch's validation loss is less than the
    previous least loss, then save the model state.
    """

    def __init__(self, save_dir: Path, best_valid_loss: float = float("inf")):
        """Save best model init."""
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir

    def __call__(self, model, epoch, current_valid_loss):
        """Save the model if it has the best val loss."""
        if current_valid_loss < self.best_valid_loss:
            file_name = f"unet_epoch_{epoch}_{current_valid_loss:.4f}.pt"
            save_path = self.save_dir / file_name
            self.best_valid_loss = current_valid_loss
            logger.info(
                f"\nBest validation loss so far: {self.best_valid_loss}"
            )
            logger.info(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save(model.state_dict(), save_path)
