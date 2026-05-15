"""Solar panel image classification with VGG16 transfer learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import kagglehub
import matplotlib.pyplot as plt
import numpy as np


class _MLPForecaster(nn.Module):
    """MLP forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, output_size: int = 90):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(90),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 2, batch_size: int = 32,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 3) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def main():
    # Download dataset
    path = kagglehub.dataset_download("pythonafroz/solar-panel-images")
    print("Path to dataset files:", path)

    # Set image dimensions
    img_height, img_width = 244, 244

    # Load and split dataset
    train_ds = utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )

    val_ds = utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )

    class_names = train_ds.class_names

    # Create model
    base_model = applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False

    # Build model architecture
    inputs = Input(shape=(img_height, img_width, 3))
    x = applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = nn.GlobalAveragePooling2D()(x)
    x = nn.Dropout(0.3)(x)
    outputs = nn.Dense(90)(x)
    model = Model(inputs, outputs)

    # Initial training
    ,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-2, patience=3, verbose=1, restore_best_weights=True
    )

    _train_torch(model, train_ds, val_ds)  # Set epochs as 2, should be more

    # Fine tuning
    base_model.trainable = True
    for layer in base_model.layers[:14]:
        layer.trainable = False

    ,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = _train_torch(model, train_ds, val_ds)

    # Plot training results
    metrics = ["accuracy", "loss"]
    for metric in metrics:
        plt.figure()
        plt.plot(history.history[metric], "g", label=f"Training {metric}")
        plt.plot(history.history[f"val_{metric}"], "r", label=f"Validation {metric}")
        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.legend()

    # Evaluate model
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation accuracy: {accuracy:.2f}")

    # Show predictions
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predictions = _predict_torch(model, images[i].unsqueeze(0))
            predicted_class = np.argmax(torch.softmax(predictions[0], dim=-1))

            color = (
                "green" if class_names[labels[i]] == class_names[predicted_class] else "red"
            )
            plt.title(f"Actual: {class_names[labels[i]]}")
            plt.ylabel(
                f"Predicted: {class_names[predicted_class]}", fontdict={"color": color}
            )
            plt.gca().axes.set_xticklabels([])
            plt.gca().axes.set_yticklabels([])

    plt.show()


if __name__ == "__main__":
    main()
