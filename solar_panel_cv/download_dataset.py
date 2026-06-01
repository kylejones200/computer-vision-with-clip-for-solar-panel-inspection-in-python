"""Auto-split from legacy monolithic script."""

import kagglehub
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import CLIPModel, CLIPProcessor


def download_dataset() -> None:
    path = kagglehub.dataset_download("pythonafroz/solar-panel-images")
    print("Path to dataset files:", path)
    img_height, img_width = (224, 224)
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
    correct = 0
    total = 0
    for images, labels in val_ds:
        predictions = predict_clip(images)
        predicted_classes = predictions.argmax(dim=-1).numpy()
        correct += (predicted_classes == labels.numpy()).sum()
        total += len(labels)

    accuracy = correct / total
    print(f"Validation accuracy: {accuracy:.2f}")
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        predictions = predict_clip(images)
        predicted_classes = predictions.argmax(dim=-1).numpy()
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            color = "green" if labels[i] == predicted_classes[i] else "red"
            plt.title(f"Actual: {class_names[labels[i]]}")
            plt.ylabel(
                f"Predicted: {class_names[predicted_classes[i]]}",
                fontdict={"color": color},
            )
            plt.gca().axes.set_xticklabels([])
            plt.gca().axes.set_yticklabels([])

    plt.show()
