"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from transformers import CLIPModel, CLIPProcessor


def set_up_binary_classification_directories() -> None:
    binary_dataset_path = "/content/binary_solar_panels/"
    img_height, img_width = (224, 224)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        binary_dataset_path,
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        binary_dataset_path,
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
        shuffle=True,
    )
    class_names = train_ds.class_names
    print("Classes:", class_names)
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    y_true = []
    y_pred_probs = []
    for images, labels in val_ds:
        predictions = predict_clip(images)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(predictions[:, 1])

    y_pred_probs = np.array(y_pred_probs)
    y_true = np.array(y_true)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    y_pred = (y_pred_probs > optimal_threshold).astype(int)
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
    plt.figure(figsize=(8, 6))
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
    plot_predictions(val_ds)
    plt.figure(figsize=(10, 6))
    clean_probs = y_pred_probs[y_true == 0]
    not_clean_probs = y_pred_probs[y_true == 1]
    plt.hist(clean_probs, alpha=0.5, label="Clean", bins=20, density=True)
    plt.hist(not_clean_probs, alpha=0.5, label="Not Clean", bins=20, density=True)
    plt.axvline(
        x=optimal_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold ({optimal_threshold:.3f})",
    )
    plt.xlabel("Probability of Not Clean Class")
    plt.ylabel("Density")
    plt.title("Distribution of CLIP Probabilities")
    plt.legend()
    plt.show()
    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.3f}")
    print(f"AUC-ROC: {roc_auc:.3f}")
