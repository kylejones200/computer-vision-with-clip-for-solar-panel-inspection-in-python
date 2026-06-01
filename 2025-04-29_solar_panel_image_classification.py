"""Solar panel image classification demo (PyTorch, synthetic fallback)."""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))


def synthetic_image_batch(n: int = 64, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    images = rng.normal(0.5, 0.2, (n, size, size, 3)).astype(np.float32)
    labels = rng.integers(0, 2, size=n)
    images[labels == 1] += 0.25
    return np.clip(images, 0, 1), labels


def train_demo() -> None:
    images, labels = synthetic_image_batch(n=128)
    X = torch.from_numpy(images.transpose(0, 3, 1, 2))
    y = torch.from_numpy(labels.astype(np.int64))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    model = SimpleCNN(num_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(5):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(X[:8]).argmax(dim=1).numpy()
    logger.info("Sample predictions: %s", preds)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(images[0])
    ax.set_title("Synthetic solar panel patch")
    plt.savefig("solar_panel_demo.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved solar_panel_demo.png")


def main() -> None:
    try:
        import kagglehub  # noqa: F401

        logger.info("kagglehub available; using synthetic demo for offline reproducibility")
    except ImportError:
        logger.info("kagglehub not installed; using synthetic image demo")
    train_demo()


if __name__ == "__main__":
    main()
