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

def download_dataset_2() -> None:
    path = "/content/a/Faulty_solar_panel/"

    img_height, img_width = (224, 224)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )

    class_names = train_ds.class_names

    num_classes = len(class_names)

    model = create_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    epochs = 20

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=5,
        verbose=1,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping]
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)

    plt.plot(history.history["accuracy"], label="Train Accuracy")

    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(history.history["loss"], label="Train Loss")

    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.show()

    test_loss, test_accuracy = model.evaluate(val_ds)

    print(f"Test accuracy: {test_accuracy:.2f}")

    plot_images_with_predictions(val_ds)

    y_true = []

    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))

