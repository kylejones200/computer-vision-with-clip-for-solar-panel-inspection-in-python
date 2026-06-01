"""Auto-split from legacy monolithic script."""

import tensorflow as tf
from sklearn.metrics import classification_report
from transformers import CLIPModel, CLIPProcessor


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
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
