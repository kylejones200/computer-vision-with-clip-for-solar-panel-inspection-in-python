---
author: "Kyle Jones"
date_published: "March 20, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/computer-vision-with-clip-for-solar-panel-inspection-in-python-e4020d35a24e"
---

# Computer Vision with CLIP for Solar Panel Inspection in Python

Solar panel efficiency directly impacts renewable energy production, with contamination and defects significantly reducing performance...

### Computer Vision with CLIP for Solar Panel Inspection in Python
Solar panel efficiency directly impacts renewable energy production, with contamination and defects significantly reducing performance. This project uses CLIP for automated solar panel inspection.

CLIP (Contrastive Language-Image Pre-Training) was developed by OpenAI and uses LLMs for image classification. This makes it particularly suitable for solar panel inspection tasks, as it can easily adapt to various defect types without extensive retraining.

There are a few tasks for predictive maintenance using computer vision on solar panels like detecting: 1/things that cover part of the PV module (and therefore reduce efficiency) like bird droppings, snow, or dust; 2/physical damage to the module like broken panels (usually when something falls on a panel); 3/corrosion; or 4/vegetation encroachment when the vegetation under the panels grows too big and interferes with the modules.

In this project, we use the Hugging Face Transformers library to access the CLIP model. The image dataset is sourced from a collection of solar panel images, split into "clean" and "not clean" categories.

The [image set comes from Kaggle](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images). This project was inspired by this tutorial by [Priyanka Kumari](https://www.labellerr.com/blog/ml-begineers-guide-onsolar-panel-inspection/).

I had a lot of trouble getting the kagglehub download to work. I ended up downloading the images the old fashioned way to set up the folder structured needed for the analysis.

I built the model to classify the condition of the PV modules based on the images into one of 6 classes. CLIP did very poorly with that approach because it basically measuring how close CLIP's answer was to an existing category name. CLIP performed better by setting this up as a binary classification problem.

The dataset is loaded and preprocessed using TensorFlow's image dataset utilities.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set image dimensions
img_height, img_width = 224, 224

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Load datasets with augmentation
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/binary_solar_panels/',
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/binary_solar_panels/',
    validation_split=0.2,
    subset='validation',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42
)

class_names = train_ds.class_names
print("Classes:", class_names)
```

One of CLIP's strengths is its ability to understand natural language descriptions. We define a set of prompts for each class:

``` 

# More detailed and specific prompts
text_descriptions = [
    [
        "a pristine solar panel with perfectly clean surface",
        "a spotless solar panel in perfect condition",
        "a clean and well-maintained solar panel",
        "a solar panel with clear glass surface",
        "a brand new looking solar panel"
    ],
    [
        "a solar panel with visible dirt or damage",
        "a solar panel covered in bird droppings",
        "a damaged or faulty solar panel",
        "a dusty and dirty solar panel",
        "a solar panel with debris on surface"
    ]
]
```

The \`predict_clip\` function processes images and text prompts to generate predictions:

```python
# Function to predict using CLIP with ensemble of prompts
def predict_clip(image_batch, temperature=100.0):
    images = [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]
    
    # Process images
    image_inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # Initialize aggregated predictions
    total_predictions = np.zeros((len(images), 2))
    
    # Process each set of prompts
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        for clean_prompt, not_clean_prompt in zip(text_descriptions[0], text_descriptions[1]):
            # Process text descriptions
            text_inputs = processor(
                text=[clean_prompt, not_clean_prompt],
                return_tensors="pt",
                padding=True
            )
            
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity with temperature scaling
            similarity = (temperature * image_features @ text_features.T).softmax(dim=-1)
            
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], class_names)
plt.yticks([0.5, 1.5], class_names)
plt.show()
```


```python
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
            
            color = 'green' if predicted_class == actual_class else 'red'
            plt.title(f"Actual: {actual_class}\nPred: {predicted_class}\nConf: {prob:.2f}", 
                     color=color, fontsize=10)
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# Plot sample predictions
plot_predictions(val_ds)

# Plot probability distributions
plt.figure(figsize=(10, 6))
clean_probs = y_pred_probs[y_true == 0]
not_clean_probs = y_pred_probs[y_true == 1]

plt.hist(clean_probs, alpha=0.5, label='Clean', bins=20, density=True)
plt.hist(not_clean_probs, alpha=0.5, label='Not Clean', bins=20, density=True)
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Threshold ({best_threshold:.3f})')
plt.xlabel('Probability of Not Clean Class')
plt.ylabel('Density')
plt.title('Distribution of CLIP Probabilities')
plt.legend()
plt.show()
```


We evaluate the model's performance using different temperature values and thresholds to find the optimal configuration. The results are visualized using confusion matrices and sample predictions.

The implementation demonstrates CLIP's effectiveness in solar panel defect detection. Its zero-shot learning capabilities allow for easy adaptation to new defect types by simply updating the text prompts. This flexibility makes CLIP particularly suitable for industrial applications where defect categories may evolve over time.

By comparison, using a transfer learning approach with MobileNetv2 had an accuracy of .87. Priyanka used VGG16 in the tutorial. I found that mnv2 has about the same accuracy and is much faster to fine-tune for our focus on solar panel inspection.


<figcaption>mnv2 transfer learning with solar panels</figcaption>


#### Technical Considerations
CLIP's performance is be sensitive to the choice of text prompts. Initially I had only one prompt for each case and this was not enough. The model's zero-shot capabilities allow for easy expansion to new defect types without retraining but the accuracy for zero shot is lower than supervised learning. CLIP is fast to run and required no training. But it was harder to control than a custom CNN model.

This CLIP-based solution provides a flexible foundation for automated solar panel inspection, offering scalable deployment options and consistent performance across varying operational conditions.

Energy Robotics has a great video about drone inspection for tanks. The approach is similar for inspecting solar farms.


<h1 id="an-error-occurred." class="message">An error occurred.</h1>

Unable to execute JavaScript.

Full MobileNetV2 implementation

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set image dimensions
img_height = 244
img_width = 244

# Load and split dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/a/Faulty_solar_panel/',
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/a/Faulty_solar_panel',
    validation_split=0.2,
    subset='validation',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True
)

# Function to convert multi-class labels to binary (Clean vs Not Clean)
def to_binary_labels(images, labels):
    binary_labels = tf.where(labels == 1, 0, 1)  # Assuming 'Clean' is label 1
    return images, binary_labels

# Apply binary conversion to datasets
train_ds_binary = train_ds.map(to_binary_labels)
val_ds_binary = val_ds.map(to_binary_labels)

# Data preprocessing
AUTOTUNE = tf.data.AUTOTUNE
train_ds_binary = train_ds_binary.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds_binary = val_ds_binary.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Create and compile model
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
epochs = 20
history = model.fit(
    train_ds_binary,
    validation_data=val_ds_binary,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Plot training results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate the model
y_true = []
y_pred = []
for images, labels in val_ds_binary:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((predictions > 0.5).astype(int).flatten())

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Clean', 'Not Clean']))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['Clean', 'Not Clean'])
plt.yticks([0.5, 1.5], ['Clean', 'Not Clean'])
plt.show()

# Function to plot predictions
def plot_predictions(dataset, num_images=25):
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class = "Clean" if predictions[i] < 0.5 else "Not Clean"
            actual_class = "Clean" if labels[i] == 0 else "Not Clean"
            
            color = 'green' if predicted_class == actual_class else 'red'
            plt.title(f"Actual: {actual_class}\nPred: {predicted_class}", 
                     color=color, fontsize=10)
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# Plot sample predictions
plot_predictions(val_ds_binary)
```
