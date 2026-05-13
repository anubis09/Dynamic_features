import copy
import logging
import time
from typing import Callable, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,  # type: ignore
        loss: Callable,
    ) -> None:
        """Initializes the Trainer and moves the model to the available device.

        Args:
            model: The neural network to train.
            optimizer: The optimizer used to update model parameters.
            loss: The loss function applied to predictions and targets.

        Attributes:
            train_losses: Average training loss per epoch, populated after calling train().
            eval_losses: Average validation loss per epoch, populated after calling train() with eval_dataloader.
            epoch_times: Wall-clock duration in seconds for each training epoch, populated after calling train().
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"Device found: {self.device}")
        self.train_losses: list[float] = []
        self.eval_losses: list[float] = []
        self.epoch_times: list[float] = []

    def train(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        patience: int,
        min_delta: float,
        eval_dataloader: Union[DataLoader, None] = None,
    ) -> None:
        """Trains the model, optionally with early stopping based on validation loss.

        Populates self.train_losses with the average training loss per epoch.
        If eval_dataloader is provided, also populates self.eval_losses and restores
        the best weights (lowest validation loss) at the end of training.

        Args:
            train_dataloader: DataLoader for the training set.
            n_epochs: Maximum number of training epochs.
            eval_dataloader: Optional DataLoader for the validation set. If provided,
                early stopping and best-weight restoration are enabled.
            patience: Number of consecutive epochs without validation loss improvement
                before stopping early. Only used when eval_dataloader is provided.
            min_delta: Minimum improvement in validation loss to be considered an
                improvement and reset the patience counter.
        """
        self.model.train()
        self.model.to(self.device)
        self.train_losses = []
        self.eval_losses = []
        self.epoch_times = []
        use_early_stopping = eval_dataloader is not None
        n_train_batches = len(train_dataloader)
        best_loss = torch.inf
        best_weights = None
        patience_count = 0

        for epoch in range(n_epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            for X, y in train_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self.epoch_times.append(time.perf_counter() - epoch_start)
            avg_train_loss = epoch_loss / n_train_batches
            logger.info(f"avg loss at epoch {epoch} is: {avg_train_loss:7f}")
            self.train_losses.append(avg_train_loss)

            if use_early_stopping:
                eval_loss = self.evaluate(eval_dataloader)
                self.eval_losses.append(eval_loss)
                logger.info(f"eval loss at epoch {epoch} is: {eval_loss:.7f}")
                self.model.train()
                if eval_loss + min_delta < best_loss:
                    best_loss = eval_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best weights.")
                        break

        if use_early_stopping and best_weights is not None:
            self.model.load_state_dict(best_weights)
        self.model.to("cpu")
        self.model.eval()

    def evaluate(self, dataloader: DataLoader) -> float:
        """Computes the average loss over a dataloader without updating model weights.

        Sets the model to eval mode before inference and leaves it in eval mode
        after returning.

        Args:
            dataloader: DataLoader to evaluate on.

        Returns:
            Average loss per batch across the full dataloader.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                total_loss += self.loss(preds, y).item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss
