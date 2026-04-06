from __future__ import annotations

from src.models.deep_learning import DeepLearningTrainer


class Trainer(DeepLearningTrainer):
    def __init__(self, model, device="cpu", epochs: int = 10, patience: int = 3):
        super().__init__(model=model, device=device)
        self.epochs = epochs
        self.patience = patience

    def train(self, train_loader, val_loader):
        return super().train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            patience=self.patience,
        )
