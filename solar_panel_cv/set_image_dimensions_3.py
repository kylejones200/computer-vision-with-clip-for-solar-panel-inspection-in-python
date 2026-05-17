"""Auto-split from legacy monolithic script."""

import os
import shutil
import warnings
from zipfile import ZipFile
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from tensorflow.keras.utils import plot_model
from transformers import CLIPModel, CLIPProcessor

def set_image_dimensions_3() -> None:
    img_height = 244

    img_width = 244

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "/content/a/Faulty_solar_panel/",
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "/content/a/Faulty_solar_panel",
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
        shuffle=True,
    )

    train_ds_binary = train_ds.map(to_binary_labels)

    val_ds_binary = val_ds.map(to_binary_labels)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_binary = (
        train_ds_binary.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    )

    val_ds_binary = val_ds_binary.cache().prefetch(buffer_size=AUTOTUNE)

    model = create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    epochs = 20

    history = model.fit(
        train_ds_binary,
        validation_data=val_ds_binary,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)

    plt.plot(history.history["accuracy"], label="Training Accuracy")

    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(history.history["loss"], label="Training Loss")

    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.show()

    y_true = []

    y_pred = []

    for images, labels in val_ds_binary:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((predictions > 0.5).astype(int).flatten())

    print("\nClassification Report:")

    print(classification_report(y_true, y_pred, target_names=["Clean", "Not Clean"]))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.xticks([0.5, 1.5], ["Clean", "Not Clean"])

    plt.yticks([0.5, 1.5], ["Clean", "Not Clean"])

    plt.show()

    plot_predictions(val_ds_binary)

