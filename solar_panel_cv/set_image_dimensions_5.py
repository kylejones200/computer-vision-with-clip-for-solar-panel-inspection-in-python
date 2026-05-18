"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from transformers import CLIPModel, CLIPProcessor


def set_image_dimensions_5() -> None:
    img_height, img_width = (224, 224)
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ]
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "/content/binary_solar_panels/",
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "/content/binary_solar_panels/",
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )
    class_names = train_ds.class_names
    print("Classes:", class_names)
    temperatures = [50.0, 100.0, 150.0]
    best_accuracy = 0
    best_temperature = None
    best_threshold = None
    for temp in temperatures:
        print(f"\nTesting temperature: {temp}")
        y_true = []
        y_pred_probs = []
        for images, labels in val_ds:
            predictions = predict_clip(images, temperature=temp)
            y_true.extend(labels.numpy())
            y_pred_probs.extend(predictions[:, 1])
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        thresholds = np.arange(0.3, 0.7, 0.05)
        for threshold in thresholds:
            y_pred = (y_pred_probs > threshold).astype(int)
            accuracy = np.mean(y_pred == y_true)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_temperature = temp
                best_threshold = threshold

    print(f"\nBest temperature: {best_temperature}")
    print(f"Best threshold: {best_threshold}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    y_true = []
    y_pred_probs = []
    for images, labels in val_ds:
        predictions = predict_clip(images, temperature=best_temperature)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(predictions[:, 1])

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > best_threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks([0.5, 1.5], class_names)
    plt.yticks([0.5, 1.5], class_names)
    plt.show()
    plot_predictions(val_ds)
    plt.figure(figsize=(10, 6))
    clean_probs = y_pred_probs[y_true == 0]
    not_clean_probs = y_pred_probs[y_true == 1]
    plt.hist(clean_probs, alpha=0.5, label="Clean", bins=20, density=True)
    plt.hist(not_clean_probs, alpha=0.5, label="Not Clean", bins=20, density=True)
    plt.axvline(
        x=best_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold ({best_threshold:.3f})",
    )
    plt.xlabel("Probability of Not Clean Class")
    plt.ylabel("Density")
    plt.title("Distribution of CLIP Probabilities")
    plt.legend()
    plt.show()
