"""
Compatibility dataset utilities for deep-learning scripts.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.models.deep_learning import (
    Vocabulary,
    SuicidalIdeationDataset,
    collate_fn,
)


def build_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    vocab: Vocabulary,
    batch_size: int = 32,
    max_len: int = 128,
):
    train_ds = SuicidalIdeationDataset(X_train, list(y_train), vocab=vocab, max_len=max_len)
    val_ds = SuicidalIdeationDataset(X_val, list(y_val), vocab=vocab, max_len=max_len)
    test_ds = SuicidalIdeationDataset(X_test, list(y_test), vocab=vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
