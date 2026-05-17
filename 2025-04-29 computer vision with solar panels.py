"""Generated from Jupyter notebook: 2025-04-29 computer vision with solar panels

Magics and shell lines are commented out. Run with a normal Python interpreter."""

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


def download_dataset() -> None:
    path = kagglehub.dataset_download("pythonafroz/solar-panel-images")

    print("Path to dataset files:", path)

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
            ax = plt.subplot(4, 4, i + 1)
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


def set_image_dimensions() -> None:
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

    class_names = train_ds.class_names

    print(f"Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = create_model(len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    epochs = 20

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping]
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

    for images, labels in val_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    print("\nClassification Report:")

    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_predictions(val_ds)


def notebook_step_009() -> None:
    zf = ZipFile("archive (1).zip", "r")

    zf.extractall("a")

    zf.close()


def set_image_dimensions_2() -> None:
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

    class_names = train_ds.class_names

    print(f"Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_mobile = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    val_ds_mobile = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    mobile_model = create_mobilenet_model(len(class_names))

    mobile_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Training MobileNetV2 Model...")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history_mobile = mobile_model.fit(
        train_ds_mobile,
        validation_data=val_ds_mobile,
        epochs=20,
        callbacks=[early_stopping],
    )

    y_true, y_pred_mobile, y_pred_clip = evaluate_both_models(val_ds, class_names)

    print("\nMobileNetV2 Classification Report:")

    print(classification_report(y_true, y_pred_mobile, target_names=class_names))

    print("\nCLIP Classification Report:")

    print(classification_report(y_true, y_pred_clip, target_names=class_names))

    plot_training_history(history_mobile)

    plot_comparison_predictions(val_ds)


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


def download_dataset_3() -> None:
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
            ax = plt.subplot(4, 4, i + 1)
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


def set_image_dimensions_4() -> None:
    img_height, img_width = (224, 224)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "/content/a/Faulty_solar_panel/",
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "/content/a/Faulty_solar_panel/",
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=32,
        seed=42,
    )

    original_class_names = train_ds.class_names

    print("Original classes:", original_class_names)

    val_ds_binary = val_ds.map(to_binary_labels)

    text_descriptions = [
        "a pristine clean solar panel with clear surface, no dirt, no damage, perfect condition",
        "a problematic solar panel with issues such as dirt, damage, bird droppings, snow, or electrical faults",
    ]

    y_true = []

    y_pred_probs = []

    for images, labels in val_ds_binary:
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

    print(classification_report(y_true, y_pred, target_names=["Clean", "Not Clean"]))

    plt.figure(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.xticks([0.5, 1.5], ["Clean", "Not Clean"])

    plt.yticks([0.5, 1.5], ["Clean", "Not Clean"])

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

    plot_predictions(val_ds_binary)

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


def set_up_binary_classification_directories() -> None:
    source_path = "/content/a/Faulty_solar_panel/"

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

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_descriptions = [
        "a pristine clean solar panel with clear surface, no dirt, no damage, perfect condition",
        "a problematic solar panel with issues such as dirt, damage, bird droppings, snow, or electrical faults",
    ]

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


def set_image_dimensions_5() -> None:
    img_height, img_width = (224, 224)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    data_augmentation = tf.keras.Sequential(
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

    text_descriptions = [
        [
            "a pristine solar panel with perfectly clean surface",
            "a spotless solar panel in perfect condition",
            "a clean and well-maintained solar panel",
            "a solar panel with clear glass surface",
            "a brand new looking solar panel",
        ],
        [
            "a solar panel with visible dirt or damage",
            "a solar panel covered in bird droppings",
            "a damaged or faulty solar panel",
            "a dusty and dirty solar panel",
            "a solar panel with debris on surface",
        ],
    ]

    temperatures = [50.0, 100.0, 150.0]

    best_accuracy = 0

    best_temperature = None

    best_threshold = None

    best_predictions = None

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
                best_predictions = y_pred

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


def set_image_dimensions_6() -> None:
    img_height, img_width = (224, 224)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

    class_counts = np.zeros(2)

    for _, labels in val_ds:
        for label in labels:
            class_counts[label.numpy()] += 1

    total_samples = np.sum(class_counts)

    class_weights = total_samples / (2 * class_counts)

    print("\nClass distribution:")

    print(f"Clean: {class_counts[0]} images")

    print(f"Not Clean: {class_counts[1]} images")

    print("\nClass weights:")

    print(f"Clean: {class_weights[0]:.2f}")

    print(f"Not Clean: {class_weights[1]:.2f}")

    text_descriptions = {
        "clean": [
            "a perfectly clean solar panel with pristine glass surface",
            "a brand new solar panel with crystal clear surface",
            "a spotless solar panel with mirror-like surface",
            "a professionally cleaned solar panel with perfect clarity",
            "a solar panel with completely clear glass",
        ],
        "not_clean": [
            "a solar panel covered with visible dirt and debris",
            "a damaged solar panel with visible defects",
            "a solar panel with bird droppings and contamination",
            "a solar panel with visible wear and damage",
            "a dirty and contaminated solar panel",
        ],
    }

    print("\nEvaluating model...")

    y_true = []

    y_pred_probs = []

    for images, labels in val_ds:
        predictions = predict_clip_with_confidence(images)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(predictions)

    y_true = np.array(y_true)

    y_pred_probs = np.array(y_pred_probs)

    thresholds = np.arange(0.3, 0.7, 0.05)

    best_f1 = 0

    best_threshold = None

    for threshold in thresholds:
        y_pred = (y_pred_probs[:, 1] > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold:.3f}")

    y_pred = (y_pred_probs[:, 1] > best_threshold).astype(int)

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

    clean_probs = y_pred_probs[y_true == 0][:, 1]

    not_clean_probs = y_pred_probs[y_true == 1][:, 1]

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

    plt.title("Distribution of Predictions")

    plt.legend()

    plt.show()


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


if __name__ == "__main__":
    main()
