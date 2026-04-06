from __future__ import annotations


class TransformerTrainer:
    """
    Lightweight compatibility wrapper around TransformerClassifier.train().
    """

    def __init__(
        self,
        classifier,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
    ):
        self.classifier = classifier
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio

    def train(self, X_train, y_train, X_val, y_val):
        self.classifier.train(
            train_texts=list(X_train),
            train_labels=list(y_train),
            val_texts=list(X_val),
            val_labels=list(y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
        )
        return {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_macro_f1": [],
        }
