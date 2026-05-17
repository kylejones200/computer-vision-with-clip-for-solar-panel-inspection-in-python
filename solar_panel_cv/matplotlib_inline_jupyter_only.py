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

def matplotlib_inline_jupyter_only() -> None:
    warnings.filterwarnings("ignore")

    img_height = 244

    img_width = 244

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "/kaggle/input/solar-panel-images/Faulty_solar_panel",
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "/kaggle/input/solar-panel-images/Faulty_solar_panel",
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
        shuffle=True,
    )

    class_names = train_ds.class_names

    print(class_names)

    train_ds

    plt.figure(figsize=(15, 15))

    for images, labels in train_ds.take(1):
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

