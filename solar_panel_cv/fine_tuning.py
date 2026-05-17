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

def fine_tuning() -> None:
    base_model.trainable = True

    for layer in base_model.layers[:14]:
        layer.trainable = False

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    epoch = 15

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoch,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0.01, patience=3, verbose=1
            )
        ],
    )

    get_ac = history.history["accuracy"]

    get_los = history.history["loss"]

    val_acc = history.history["val_accuracy"]

    val_loss = history.history["val_loss"]

    epochs = range(len(get_ac))

    plt.plot(epochs, get_ac, "g", label="Accuracy of Training data")

    plt.plot(epochs, get_los, "r", label="Loss of Training data")

    plt.title("Training data accuracy and loss")

    plt.figure()

    plt.plot(epochs, get_ac, "g", label="Accuracy of Training Data")

    plt.plot(epochs, val_acc, "r", label="Accuracy of Validation Data")

    plt.title("Training and Validation Accuracy")

    plt.figure()

    plt.plot(epochs, get_los, "g", label="Loss of Training Data")

    plt.plot(epochs, val_loss, "r", label="Loss of Validation Data")

    plt.title("Training and Validation Loss")

    plt.figure()

    plt.show()

    loss, accuracy = model.evaluate(val_ds)

    plt.figure(figsize=(20, 20))

    for images, labels in val_ds.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predictions = model.predict(tf.expand_dims(images[i], 0))
            score = tf.nn.softmax(predictions[0])
            if class_names[labels[i]] == class_names[np.argmax(score)]:
                plt.title("Actual: " + class_names[labels[i]])
                plt.ylabel(
                    "Predicted: " + class_names[np.argmax(score)],
                    fontdict={"color": "green"},
                )
            else:
                plt.title("Actual: " + class_names[labels[i]])
                plt.ylabel(
                    "Predicted: " + class_names[np.argmax(score)],
                    fontdict={"color": "red"},
                )
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticklabels([])

