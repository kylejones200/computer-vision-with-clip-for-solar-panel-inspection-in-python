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

def create_mobilenet_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes),
        ]
    )
    return model


def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def evaluate_both_models(val_ds, class_names):
    y_true = []
    y_pred_mobile = []
    y_pred_clip = []
    for images, labels in val_ds:
        mobile_pred = mobile_model.predict(images)
        y_pred_mobile.extend(np.argmax(mobile_pred, axis=1))
        clip_pred = get_clip_predictions(images, class_names)
        y_pred_clip.extend(np.argmax(clip_pred, axis=1))
        y_true.extend(labels.numpy())
    return (y_true, y_pred_mobile, y_pred_clip)


def get_clip_predictions(images, class_names):
    images_pil = [Image.fromarray(img.numpy().astype("uint8")) for img in images]
    text_inputs = clip_processor(text=class_names, return_tensors="pt", padding=True)
    image_inputs = clip_processor(images=images_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity.numpy()


def plot_comparison_predictions(val_ds, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        mobile_predictions = mobile_model.predict(images)
        clip_predictions = get_clip_predictions(images, class_names)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            mobile_pred = np.argmax(mobile_predictions[i])
            clip_pred = np.argmax(clip_predictions[i])
            actual_class = labels[i].numpy()
            mobile_correct = mobile_pred == actual_class
            clip_correct = clip_pred == actual_class
            title = f"Actual: {class_names[actual_class]}\n"
            title += f"MobileNet: {class_names[mobile_pred]} ({('✓' if mobile_correct else '✗')})\n"
            title += (
                f"CLIP: {class_names[clip_pred]} ({('✓' if clip_correct else '✗')})"
            )
            plt.title(title, fontsize=10)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_images_with_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class = np.argmax(predictions[i])
            actual_class = labels[i].numpy()
            color = "green" if predicted_class == actual_class else "red"
            plt.title(
                f"Actual: {class_names[actual_class]}\nPredicted: {class_names[predicted_class]}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = predict_clip_with_confidence(images)
        predicted_classes = (predictions[:, 1] > best_threshold).astype(int)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            pred_class = class_names[predicted_classes[i]]
            true_class = class_names[labels[i]]
            prob = predictions[i][predicted_classes[i]]
            color = "green" if pred_class == true_class else "red"
            plt.title(
                f"True: {true_class}\nPred: {pred_class}\nConf: {prob:.2f}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("MobileNetV2 Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("MobileNetV2 Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def predict_clip(image_batch, temperature=100.0):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]
    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    total_predictions = np.zeros((len(images), 2))
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for clean_prompt, not_clean_prompt in zip(
            text_descriptions[0], text_descriptions[1]
        ):
            text_inputs = processor(
                text=[clean_prompt, not_clean_prompt], return_tensors="pt", padding=True
            )
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (temperature * image_features @ text_features.T).softmax(
                dim=-1
            )
            total_predictions += similarity.numpy()
    return total_predictions / len(text_descriptions[0])


def predict_clip_with_confidence(image_batch, temperature=50.0):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]
    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    clean_scores = []
    not_clean_scores = []
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for clean_prompt, not_clean_prompt in zip(
            text_descriptions["clean"], text_descriptions["not_clean"]
        ):
            text_inputs = processor(
                text=[clean_prompt, not_clean_prompt], return_tensors="pt", padding=True
            )
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = temperature * image_features @ text_features.T
            clean_scores.append(similarity[:, 0])
            not_clean_scores.append(similarity[:, 1])
    clean_scores = torch.stack(clean_scores).mean(dim=0) * class_weights[0]
    not_clean_scores = torch.stack(not_clean_scores).mean(dim=0) * class_weights[1]
    scores = torch.stack([clean_scores, not_clean_scores], dim=1)
    probabilities = torch.softmax(scores, dim=1)
    return probabilities.numpy()


def setup_binary_dataset(source_path, dest_path):
    os.makedirs(os.path.join(dest_path, "Clean"), exist_ok=True)
    os.makedirs(os.path.join(dest_path, "Not_Clean"), exist_ok=True)
    classes = os.listdir(source_path)
    for class_name in classes:
        source_class_path = os.path.join(source_path, class_name)
        if not os.path.isdir(source_class_path):
            continue
        if class_name == "Clean":
            dest_dir = os.path.join(dest_path, "Clean")
        else:
            dest_dir = os.path.join(dest_path, "Not_Clean")
        for img_name in os.listdir(source_class_path):
            source_file = os.path.join(source_class_path, img_name)
            new_filename = f"{class_name}_{img_name}"
            dest_file = os.path.join(dest_dir, new_filename)
            shutil.copy2(source_file, dest_file)
    clean_count = len(os.listdir(os.path.join(dest_path, "Clean")))
    not_clean_count = len(os.listdir(os.path.join(dest_path, "Not_Clean")))
    print("Dataset created with:")
    print(f"Clean images: {clean_count}")
    print(f"Not Clean images: {not_clean_count}")
    return dest_path


def to_binary_labels(images, labels):
    clean_idx = original_class_names.index("Clean")
    binary_labels = tf.where(labels == clean_idx, 0, 1)
    return (images, binary_labels)


def notebook_step_004() -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    epoch = 15

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoch,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.01,
                patience=3,
                verbose=1,
                restore_best_weights=True,
            )
        ],
    )


def notebook_step_009() -> None:
    zf = ZipFile("archive (1).zip", "r")

    zf.extractall("a")

    zf.close()


def main() -> None:
    matplotlib_inline_jupyter_only()
    matplotlib_inline_jupyter_only_2()
    notebook_step_004()
    fine_tuning()
    download_dataset()
    download_dataset_2()
    set_image_dimensions()
    notebook_step_009()
    set_image_dimensions_2()
    set_image_dimensions_3()
    download_dataset_3()
    set_image_dimensions_4()
    set_up_binary_classification_directories()
    set_image_dimensions_5()
    set_image_dimensions_6()

