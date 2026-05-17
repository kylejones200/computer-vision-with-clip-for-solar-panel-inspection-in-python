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

def matplotlib_inline_jupyter_only_2() -> None:
    plot_model(model, to_file="cnn_plot.png", show_shapes=True, show_layer_names=True)

    warnings.filterwarnings("ignore")

    path = kagglehub.dataset_download("pythonafroz/solar-panel-images")

    print("Path to dataset files:", path)

    img_height = 244

    img_width = 244

    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
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

    base_model = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))

    x = tf.keras.applications.vgg16.preprocess_input(inputs)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(90)(x)

    model = tf.keras.Model(inputs, outputs)

    model.summary()

    plot_model(model, to_file="cnn_plot.png", show_shapes=True, show_layer_names=True)

