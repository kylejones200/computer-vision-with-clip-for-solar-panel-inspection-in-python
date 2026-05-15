"""Solar panel image classification with VGG16 transfer learning."""

import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub
import numpy as np

# Download dataset
path = kagglehub.dataset_download("pythonafroz/solar-panel-images")
print("Path to dataset files:", path)

# Set image dimensions
img_height, img_width = 244, 244

# Load and split dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset='training',
    image_size=(img_height, img_width), batch_size=32, seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset='validation',
    image_size=(img_height, img_width), batch_size=32, seed=42
)

class_names = train_ds.class_names

# Create model
base_model = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet',
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False

# Build model architecture
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(90)(x)
model = tf.keras.Model(inputs, outputs)

# Initial training
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=1e-2, patience=3, verbose=1, restore_best_weights=True
)

model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=[early_stopping])  # Set epochs as 2, should be more

# Fine tuning
base_model.trainable = True
for layer in base_model.layers[:14]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stopping])

# Plot training results
metrics = ['accuracy', 'loss']
for metric in metrics:
    plt.figure()
    plt.plot(history.history[metric], 'g', label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], 'r', label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.legend()

# Evaluate model
loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy:.2f}")

# Show predictions
plt.figure(figsize=(20, 20))
for images, labels in val_ds.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predictions = model.predict(tf.expand_dims(images[i], 0))
        predicted_class = np.argmax(tf.nn.softmax(predictions[0]))

        color = 'green' if class_names[labels[i]] == class_names[predicted_class] else 'red'
        plt.title(f"Actual: {class_names[labels[i]]}")
        plt.ylabel(f"Predicted: {class_names[predicted_class]}", fontdict={'color': color})
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])

plt.show()
