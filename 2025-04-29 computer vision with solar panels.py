"""Generated from Jupyter notebook: 2025-04-29 computer vision with solar panels

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %matplotlib inline  # Jupyter-only


img_height = 244
img_width = 244
train_ds = utils.image_dataset_from_directory(
    "/kaggle/input/solar-panel-images/Faulty_solar_panel",
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True,
)

val_ds = utils.image_dataset_from_directory(
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


# --- code cell ---

import os

# %matplotlib inline  # Jupyter-only
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plot_model(model, to_file="cnn_plot.png", show_shapes=True, show_layer_names=True)



import kagglehub



# Download latest version
path = kagglehub.dataset_download("pythonafroz/solar-panel-images")

print("Path to dataset files:", path)

img_height = 244
img_width = 244

train_ds = utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True,
)

val_ds = utils.image_dataset_from_directory(
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


base_model = applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3)
)
base_model.trainable = False

inputs = Input(shape=(img_height, img_width, 3))
x = applications.vgg16.preprocess_input(inputs)
x = base_model(x, training=False)
x = nn.GlobalAveragePooling2D()(x)
x = nn.Dropout(0.3)(x)
outputs = nn.Dense(90)(x)
model = Model(inputs, outputs)

model.summary()


plot_model(model, to_file="cnn_plot.png", show_shapes=True, show_layer_names=True)


# --- code cell ---

# !pip install pydot  # Jupyter-only
# !pip install graphviz  # Jupyter-only


# --- code cell ---

,
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
epoch = 15
_train_torch(model, train_ds, val_ds)
    ],
)


# --- code cell ---

# fine tuning
base_model.trainable = True
for layer in base_model.layers[:14]:
    layer.trainable = False
model.summary()

,
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
epoch = 15
history = _train_torch(model, train_ds, val_ds)
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
        predictions = _predict_torch(model, images[i].unsqueeze(0))
        score = torch.softmax(predictions[0], dim=-1)
        if class_names[labels[i]] == class_names[np.argmax(score)]:
            plt.title("Actual: " + class_names[labels[i]])
            plt.ylabel(
                "Predicted: " + class_names[np.argmax(score)],
                fontdict={"color": "green"},
            )

        else:
            plt.title("Actual: " + class_names[labels[i]])
            plt.ylabel(
                "Predicted: " + class_names[np.argmax(score)], fontdict={"color": "red"}
            )
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.gca().axes.xaxis.set_ticklabels([])


# --- code cell ---

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Download dataset
path = kagglehub.dataset_download("pythonafroz/solar-panel-images")
print("Path to dataset files:", path)

# Set image dimensions
img_height, img_width = 224, 224  # CLIP expects 224x224 images

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

val_ds = utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

class_names = train_ds.class_names


# Function to predict using CLIP
class _MLPForecaster(nn.Module):
    """MLP forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(90), nn.ReLU(),
            nn.Linear(90, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 32,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 3) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def predict_clip(image_batch):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]
    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Get text features for class names
    text_inputs = processor(text=class_names, return_tensors="pt", padding=True)
    text_outputs = model.get_text_features(**text_inputs)

    # Calculate similarity
    similarity = outputs @ text_outputs.T
    return similarity.softmax(dim=-1)


# Evaluate model
correct = 0
total = 0

for images, labels in val_ds:
    predictions = predict_clip(images)
    predicted_classes = predictions.argmax(dim=-1).numpy()
    correct += (predicted_classes == labels.numpy()).sum()
    total += len(labels)

accuracy = correct / total
print(f"Validation accuracy: {accuracy:.2f}")

# Show predictions
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
            f"Predicted: {class_names[predicted_classes[i]]}", fontdict={"color": color}
        )
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])

plt.show()


# --- code cell ---

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Download dataset
path = "/content/a/Faulty_solar_panel/"

# Set image dimensions
img_height, img_width = 224, 224  # CLIP expects 224x224 images

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

val_ds = utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

class_names = train_ds.class_names


# Update the model architecture
def create_model(num_classes):
    base_model = applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(img_height, img_width, 3))
    x = applications.vgg19.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = nn.GlobalAveragePooling2D()(x)
    x = nn.Dense(256, activation="relu")(x)
    x = nn.Dropout(0.5)(x)
    outputs = nn.Dense(num_classes)(x)
    return Model(inputs, outputs)


# Create and compile the model
num_classes = len(class_names)
model = create_model(num_classes)

,
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
epochs = 20
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", min_delta=1e-2, patience=5, verbose=1, restore_best_weights=True
)

history = _train_torch(model, train_ds, val_ds)

# Plot training history
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

# Evaluate the model
test_loss, test_accuracy = model.evaluate(val_ds)
print(f"Test accuracy: {test_accuracy:.2f}")


# Function to plot images with predictions
def plot_images_with_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = _predict_torch(model, images)
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


# Plot images with predictions
plot_images_with_predictions(val_ds)

# Generate classification report
y_true = []
y_pred = []
for images, labels in val_ds:
    predictions = _predict_torch(model, images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Set image dimensions
img_height = 244
img_width = 244

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    "/content/a/Faulty_solar_panel/",
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True,
)

val_ds = utils.image_dataset_from_directory(
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

# Data preprocessing
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Create the model
def create_model(num_classes):
    base_model = applications.MobileNetV2(
        input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            nn.GlobalAveragePooling2D(),
            nn.Dropout(0.2),
            nn.Dense(256, activation="relu"),
            nn.Dropout(0.5),
            nn.Dense(num_classes),
        ]
    )
    return model


# Create and compile model
model = create_model(len(class_names))
,
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train the model
epochs = 20
history = _train_torch(model, train_ds, val_ds)

# Plot training results
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


# Function to plot predictions
def plot_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = _predict_torch(model, images)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class = np.argmax(predictions[i])
            actual_class = labels[i].numpy()

            color = "green" if predicted_class == actual_class else "red"
            plt.title(
                f"Actual: {class_names[actual_class]}\nPred: {class_names[predicted_class]}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Generate predictions and classification report
y_true = []
y_pred = []
for images, labels in val_ds:
    predictions = _predict_torch(model, images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot sample predictions
plot_predictions(val_ds)


# --- code cell ---

from zipfile import ZipFile

zf = ZipFile("archive (1).zip", "r")
zf.extractall("a")
zf.close()


# --- duplicate code cell omitted (identical to earlier cell) ---


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report
from transformers import CLIPModel, CLIPProcessor

# Set image dimensions
img_height = 244
img_width = 244

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    "/content/a/Faulty_solar_panel/",
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True,
)

val_ds = utils.image_dataset_from_directory(
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

# Data preprocessing for MobileNetV2
train_ds_mobile = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds_mobile = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# 1. MobileNetV2 Model
def create_mobilenet_model(num_classes):
    base_model = applications.MobileNetV2(
        input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            nn.GlobalAveragePooling2D(),
            nn.Dropout(0.2),
            nn.Dense(256, activation="relu"),
            nn.Dropout(0.5),
            nn.Dense(num_classes),
        ]
    )
    return model


# Create and compile MobileNetV2 model
mobile_model = create_mobilenet_model(len(class_names))
,
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 2. CLIP Model Setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to get CLIP predictions
def get_clip_predictions(images, class_names):
    images_pil = [Image.fromarray(img.numpy().astype("uint8")) for img in images]

    # Prepare text inputs for all classes
    text_inputs = clip_processor(text=class_names, return_tensors="pt", padding=True)

    # Prepare image inputs
    image_inputs = clip_processor(images=images_pil, return_tensors="pt", padding=True)

    # Get predictions
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity.numpy()


# Training MobileNetV2
print("Training MobileNetV2 Model...")
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history_mobile = _train_torch(mobile_model, train_ds_mobile, val_ds_mobile)


# Evaluate both models
def evaluate_both_models(val_ds, class_names):
    # Results storage
    y_true = []
    y_pred_mobile = []
    y_pred_clip = []

    for images, labels in val_ds:
        # MobileNetV2 predictions
        mobile_pred = _predict_torch(mobile_model, images)
        y_pred_mobile.extend(np.argmax(mobile_pred, axis=1))

        # CLIP predictions
        clip_pred = get_clip_predictions(images, class_names)
        y_pred_clip.extend(np.argmax(clip_pred, axis=1))

        y_true.extend(labels.numpy())

    return y_true, y_pred_mobile, y_pred_clip


# Get predictions and evaluate
y_true, y_pred_mobile, y_pred_clip = evaluate_both_models(val_ds, class_names)

# Print classification reports
print("\nMobileNetV2 Classification Report:")
print(classification_report(y_true, y_pred_mobile, target_names=class_names))

print("\nCLIP Classification Report:")
print(classification_report(y_true, y_pred_clip, target_names=class_names))


# Plotting functions
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


def plot_comparison_predictions(val_ds, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        mobile_predictions = _predict_torch(mobile_model, images)
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
            title += f"MobileNet: {class_names[mobile_pred]} ({'✓' if mobile_correct else '✗'})\n"
            title += f"CLIP: {class_names[clip_pred]} ({'✓' if clip_correct else '✗'})"

            plt.title(title, fontsize=10)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Plot results
plot_training_history(history_mobile)
plot_comparison_predictions(val_ds)


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set image dimensions
img_height = 244
img_width = 244

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    "/content/a/Faulty_solar_panel/",
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True,
)

val_ds = utils.image_dataset_from_directory(
    "/content/a/Faulty_solar_panel",
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True,
)


# Function to convert multi-class labels to binary (Clean vs Not Clean)
def to_binary_labels(images, labels):
    binary_labels = torch.where(labels == 1, 0, 1)  # Assuming 'Clean' is label 1
    return images, binary_labels


# Apply binary conversion to datasets
train_ds_binary = train_ds.map(to_binary_labels)
val_ds_binary = val_ds.map(to_binary_labels)

# Data preprocessing
train_ds_binary = train_ds_binary.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds_binary = val_ds_binary.cache().prefetch(buffer_size=AUTOTUNE)


# Create the model
def create_model():
    base_model = applications.MobileNetV2(
        input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            nn.GlobalAveragePooling2D(),
            nn.Dropout(0.2),
            nn.Dense(256, activation="relu"),
            nn.Dropout(0.5),
            nn.Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )
    return model


# Create and compile model
model = create_model()
,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train the model
epochs = 20
history = _train_torch(model, train_ds_binary, val_ds_binary)

# Plot training results
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

# Evaluate the model
y_true = []
y_pred = []
for images, labels in val_ds_binary:
    predictions = _predict_torch(model, images)
    y_true.extend(labels.numpy())
    y_pred.extend((predictions > 0.5).astype(int).flatten())

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Clean", "Not Clean"]))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks([0.5, 1.5], ["Clean", "Not Clean"])
plt.yticks([0.5, 1.5], ["Clean", "Not Clean"])
plt.show()


# Function to plot predictions
def plot_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = _predict_torch(model, images)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class = "Clean" if predictions[i] < 0.5 else "Not Clean"
            actual_class = "Clean" if labels[i] == 0 else "Not Clean"

            color = "green" if predicted_class == actual_class else "red"
            plt.title(
                f"Actual: {actual_class}\nPred: {predicted_class}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Plot sample predictions
plot_predictions(val_ds_binary)


# --- code cell ---

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Download dataset
path = "/content/a/Faulty_solar_panel/"

# Set image dimensions
img_height, img_width = 224, 224  # CLIP expects 224x224 images

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

val_ds = utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

class_names = train_ds.class_names


# Function to predict using CLIP
def predict_clip(image_batch):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]
    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Get text features for class names
    text_inputs = processor(text=class_names, return_tensors="pt", padding=True)
    text_outputs = model.get_text_features(**text_inputs)

    # Calculate similarity
    similarity = outputs @ text_outputs.T
    return similarity.softmax(dim=-1)


# Evaluate model
correct = 0
total = 0

for images, labels in val_ds:
    predictions = predict_clip(images)
    predicted_classes = predictions.argmax(dim=-1).numpy()
    correct += (predicted_classes == labels.numpy()).sum()
    total += len(labels)

accuracy = correct / total
print(f"Validation accuracy: {accuracy:.2f}")

# Show predictions
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
            f"Predicted: {class_names[predicted_classes[i]]}", fontdict={"color": color}
        )
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])

plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from transformers import CLIPModel, CLIPProcessor

# Set image dimensions
img_height, img_width = 224, 224  # CLIP expects 224x224 images

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    "/content/a/Faulty_solar_panel/",
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

val_ds = utils.image_dataset_from_directory(
    "/content/a/Faulty_solar_panel/",
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

original_class_names = train_ds.class_names
print("Original classes:", original_class_names)


# Function to convert labels to binary (Clean vs Not Clean)
def to_binary_labels(images, labels):
    # Assuming 'Clean' is index 1 in your class names
    clean_idx = original_class_names.index("Clean")
    binary_labels = torch.where(
        labels == clean_idx, 0, 1
    )  # 0 for Clean, 1 for everything else
    return images, binary_labels


# Convert datasets to binary
val_ds_binary = val_ds.map(to_binary_labels)

# Define better prompts for CLIP
text_descriptions = [
    "a pristine clean solar panel with clear surface, no dirt, no damage, perfect condition",
    "a problematic solar panel with issues such as dirt, damage, bird droppings, snow, or electrical faults",
]


# Function to predict using CLIP with improved prompts
def predict_clip(image_batch):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]

    # Process images
    image_inputs = processor(images=images, return_tensors="pt", padding=True)

    # Process text descriptions
    text_inputs = processor(text=text_descriptions, return_tensors="pt", padding=True)

    # Get features
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity.numpy()


# Evaluate model
y_true = []
y_pred_probs = []

for images, labels in val_ds_binary:
    predictions = predict_clip(images)
    y_true.extend(labels.numpy())
    y_pred_probs.extend(predictions[:, 1])  # Probability of "not clean" class

y_pred_probs = np.array(y_pred_probs)
y_true = np.array(y_true)

# Find optimal threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.3f}")

# Make predictions using optimal threshold
y_pred = (y_pred_probs > optimal_threshold).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Clean", "Not Clean"]))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks([0.5, 1.5], ["Clean", "Not Clean"])
plt.yticks([0.5, 1.5], ["Clean", "Not Clean"])
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()


# Function to plot predictions with probabilities
def plot_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = predict_clip(images)
        predicted_classes = (predictions[:, 1] > optimal_threshold).astype(int)

        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class = "Not Clean" if predicted_classes[i] == 1 else "Clean"
            actual_class = "Not Clean" if labels[i] == 1 else "Clean"
            prob = predictions[i][1]  # Probability of "not clean"

            color = "green" if predicted_class == actual_class else "red"
            plt.title(
                f"Actual: {actual_class}\nPred: {predicted_class}\nProb(Not Clean): {prob:.2f}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Plot predictions
plot_predictions(val_ds_binary)

# Plot probability distributions
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

# Print some additional metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc:.3f}")


# --- code cell ---

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from transformers import CLIPModel, CLIPProcessor


# Set up binary classification directories
def setup_binary_dataset(source_path, dest_path):
    # Create main directories
    os.makedirs(os.path.join(dest_path, "Clean"), exist_ok=True)
    os.makedirs(os.path.join(dest_path, "Not_Clean"), exist_ok=True)

    # Get list of all classes
    classes = os.listdir(source_path)

    # Copy files to appropriate directories
    for class_name in classes:
        source_class_path = os.path.join(source_path, class_name)
        if not os.path.isdir(source_class_path):
            continue

        # Determine destination directory
        if class_name == "Clean":
            dest_dir = os.path.join(dest_path, "Clean")
        else:
            dest_dir = os.path.join(dest_path, "Not_Clean")

        # Copy files
        for img_name in os.listdir(source_class_path):
            source_file = os.path.join(source_class_path, img_name)
            # Add original class name to filename to track origin
            new_filename = f"{class_name}_{img_name}"
            dest_file = os.path.join(dest_dir, new_filename)
            shutil.copy2(source_file, dest_file)

    # Print dataset statistics
    clean_count = len(os.listdir(os.path.join(dest_path, "Clean")))
    not_clean_count = len(os.listdir(os.path.join(dest_path, "Not_Clean")))
    print("Dataset created with:")
    print(f"Clean images: {clean_count}")
    print(f"Not Clean images: {not_clean_count}")
    return dest_path


# Set paths
source_path = "/content/a/Faulty_solar_panel/"
binary_dataset_path = "/content/binary_solar_panels/"


# Set image dimensions
img_height, img_width = 224, 224  # CLIP expects 224x224 images

# Load and split dataset
train_ds = utils.image_dataset_from_directory(
    binary_dataset_path,
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True,
)

val_ds = utils.image_dataset_from_directory(
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

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define prompts for CLIP
text_descriptions = [
    "a pristine clean solar panel with clear surface, no dirt, no damage, perfect condition",
    "a problematic solar panel with issues such as dirt, damage, bird droppings, snow, or electrical faults",
]


# Function to predict using CLIP
def predict_clip(image_batch):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]

    # Process images
    image_inputs = processor(images=images, return_tensors="pt", padding=True)

    # Process text descriptions
    text_inputs = processor(text=text_descriptions, return_tensors="pt", padding=True)

    # Get features
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity.numpy()


# Evaluate model
y_true = []
y_pred_probs = []

for images, labels in val_ds:
    predictions = predict_clip(images)
    y_true.extend(labels.numpy())
    y_pred_probs.extend(predictions[:, 1])  # Probability of "not clean" class

y_pred_probs = np.array(y_pred_probs)
y_true = np.array(y_true)

# Find optimal threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.3f}")

# Make predictions using optimal threshold
y_pred = (y_pred_probs > optimal_threshold).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks([0.5, 1.5], class_names)
plt.yticks([0.5, 1.5], class_names)
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()


# Function to plot predictions
def plot_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = predict_clip(images)
        predicted_classes = (predictions[:, 1] > optimal_threshold).astype(int)

        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class = class_names[predicted_classes[i]]
            actual_class = class_names[labels[i]]
            prob = predictions[i][1]  # Probability of "not clean"

            color = "green" if predicted_class == actual_class else "red"
            plt.title(
                f"Actual: {actual_class}\nPred: {predicted_class}\nProb(Not Clean): {prob:.2f}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Plot predictions
plot_predictions(val_ds)

# Plot probability distributions
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

# Print additional metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc:.3f}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from transformers import CLIPModel, CLIPProcessor

# Set image dimensions
img_height, img_width = 224, 224

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Data augmentation
data_augmentation = Sequential(
    [
        nn.RandomFlip("horizontal"),
        nn.RandomRotation(0.1),
        nn.RandomBrightness(0.2),
        nn.RandomContrast(0.2),
    ]
)

# Load datasets with augmentation
train_ds = utils.image_dataset_from_directory(
    "/content/binary_solar_panels/",
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

val_ds = utils.image_dataset_from_directory(
    "/content/binary_solar_panels/",
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

class_names = train_ds.class_names
print("Classes:", class_names)

# More detailed and specific prompts
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


# Function to predict using CLIP with ensemble of prompts
def predict_clip(image_batch, temperature=100.0):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]

    # Process images
    image_inputs = processor(images=images, return_tensors="pt", padding=True)

    # Initialize aggregated predictions
    total_predictions = np.zeros((len(images), 2))

    # Process each set of prompts
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for clean_prompt, not_clean_prompt in zip(
            text_descriptions[0], text_descriptions[1]
        ):
            # Process text descriptions
            text_inputs = processor(
                text=[clean_prompt, not_clean_prompt], return_tensors="pt", padding=True
            )

            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity with temperature scaling
            similarity = (temperature * image_features @ text_features.T).softmax(
                dim=-1
            )

            total_predictions += similarity.numpy()

    # Average predictions across all prompt pairs
    return total_predictions / len(text_descriptions[0])


# Evaluate model with different temperature values
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

    # Try different thresholds
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

# Use best parameters for final evaluation
y_true = []
y_pred_probs = []

for images, labels in val_ds:
    predictions = predict_clip(images, temperature=best_temperature)
    y_true.extend(labels.numpy())
    y_pred_probs.extend(predictions[:, 1])

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > best_threshold).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks([0.5, 1.5], class_names)
plt.yticks([0.5, 1.5], class_names)
plt.show()


# Function to visualize predictions
def plot_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = predict_clip(images, temperature=best_temperature)
        predicted_classes = (predictions[:, 1] > best_threshold).astype(int)

        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class = class_names[predicted_classes[i]]
            actual_class = class_names[labels[i]]
            prob = predictions[i][1]

            color = "green" if predicted_class == actual_class else "red"
            plt.title(
                f"Actual: {actual_class}\nPred: {predicted_class}\nConf: {prob:.2f}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Plot sample predictions
plot_predictions(val_ds)

# Plot probability distributions
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


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import CLIPModel, CLIPProcessor

# Set image dimensions
img_height, img_width = 224, 224

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load datasets
train_ds = utils.image_dataset_from_directory(
    "/content/binary_solar_panels/",
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

val_ds = utils.image_dataset_from_directory(
    "/content/binary_solar_panels/",
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Calculate class weights
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

# Improved prompts with visual descriptions
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


def predict_clip_with_confidence(image_batch, temperature=50.0):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]

    # Process images
    image_inputs = processor(images=images, return_tensors="pt", padding=True)

    clean_scores = []
    not_clean_scores = []

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Process each prompt pair
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

    # Calculate final scores with class weights
    clean_scores = torch.stack(clean_scores).mean(dim=0) * class_weights[0]
    not_clean_scores = torch.stack(not_clean_scores).mean(dim=0) * class_weights[1]

    # Calculate probabilities
    scores = torch.stack([clean_scores, not_clean_scores], dim=1)
    probabilities = torch.softmax(scores, dim=1)

    return probabilities.numpy()


# Evaluate model
print("\nEvaluating model...")
y_true = []
y_pred_probs = []

for images, labels in val_ds:
    predictions = predict_clip_with_confidence(images)
    y_true.extend(labels.numpy())
    y_pred_probs.extend(predictions)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# Find best threshold
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

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks([0.5, 1.5], class_names)
plt.yticks([0.5, 1.5], class_names)
plt.show()


# Plot sample predictions
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


# Plot predictions
plot_predictions(val_ds)

# Plot probability distributions
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
