"""Auto-split from legacy monolithic script."""

import warnings

import matplotlib.pyplot as plt
import tensorflow as tf


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
    tf.keras.utils.image_dataset_from_directory(
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
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
