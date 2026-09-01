"""Training utilities for the DNN OTFS receiver."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class DNNTrainer:
    """Train and validate an OTFS DNN receiver."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        gradient_clip_norm: float,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        scheduler_factor: float,
        scheduler_patience: int,
        checkpoint_path: str | Path,
        history_path: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip_norm = float(gradient_clip_norm)

        self.early_stopping_patience = int(
            early_stopping_patience
        )
        self.early_stopping_min_delta = float(
            early_stopping_min_delta
        )

        self.device = device

        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = Path(history_path)

        self.criterion = nn.MSELoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(scheduler_factor),
            patience=int(scheduler_patience),
        )

        self.model.to(self.device)

    def _loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute complex-symbol reconstruction loss."""

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes do not match: "
                f"{prediction.shape} != {target.shape}."
            )

        if not torch.is_complex(prediction):
            raise TypeError(
                "Model predictions must be complex-valued."
            )

        if not torch.is_complex(target):
            raise TypeError(
                "Targets must be complex-valued."
            )

        return (
            self.criterion(
                prediction.real,
                target.real,
            )
            + self.criterion(
                prediction.imag,
                target.imag,
            )
        )

    def _train_epoch(self) -> float:
        """Run one training epoch."""

        self.model.train()

        total_loss = 0.0
        sample_count = 0

        for rx_dd, tx_dd in self.train_loader:
            rx_dd = rx_dd.to(
                self.device,
                non_blocking=True,
            )
            tx_dd = tx_dd.to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            prediction = self.model(rx_dd)

            loss = self._loss(
                prediction,
                tx_dd,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss encountered."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )

            self.optimizer.step()

            batch_size = rx_dd.shape[0]

            total_loss += (
                loss.detach().item() * batch_size
            )

            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError(
                "Training loader contains no samples."
            )

        return total_loss / sample_count

    @torch.no_grad()
    def _validate_epoch(self) -> float:
        """Run one validation epoch."""

        self.model.eval()

        total_loss = 0.0
        sample_count = 0

        for rx_dd, tx_dd in self.validation_loader:
            rx_dd = rx_dd.to(
                self.device,
                non_blocking=True,
            )
            tx_dd = tx_dd.to(
                self.device,
                non_blocking=True,
            )

            prediction = self.model(rx_dd)

            loss = self._loss(
                prediction,
                tx_dd,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite validation loss encountered."
                )

            batch_size = rx_dd.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError(
                "Validation loader contains no samples."
            )

        return total_loss / sample_count

    def _save_checkpoint(
        self,
        epoch: int,
        validation_loss: float,
    ) -> None:
        """Save the current best model state."""

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

    def _restore_best_checkpoint(self) -> None:
        """Restore the model with the best validation loss."""

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain model state."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    def _write_history(
        self,
        history: list[dict[str, Any]],
    ) -> None:
        """Write training history to CSV."""

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate",
            "best_validation_loss",
        ]

        with self.history_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(history)

    def fit(self) -> list[dict[str, Any]]:
        """Train the model with validation-based early stopping."""

        best_validation_loss = math.inf
        epochs_without_improvement = 0

        history: list[dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            validation_loss = self._validate_epoch()

            self.scheduler.step(validation_loss)

            learning_rate = float(
                self.optimizer.param_groups[0]["lr"]
            )

            improvement = (
                best_validation_loss
                - validation_loss
            )

            if improvement > self.early_stopping_min_delta:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0

                self._save_checkpoint(
                    epoch=epoch,
                    validation_loss=validation_loss,
                )
            else:
                epochs_without_improvement += 1

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": learning_rate,
                    "best_validation_loss": (
                        best_validation_loss
                    ),
                }
            )

            self._write_history(history)

            print(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"train={train_loss:.6e} | "
                f"val={validation_loss:.6e} | "
                f"lr={learning_rate:.3e}"
            )

            if (
                epochs_without_improvement
                >= self.early_stopping_patience
            ):
                print(
                    "Early stopping triggered."
                )
                break

        self._restore_best_checkpoint()

        return history