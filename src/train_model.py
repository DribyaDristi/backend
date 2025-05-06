# import tensorflow as tf

# BATCH_SIZE = 32
# IMG_SIZE = (64, 64)

# dataset_path = "C:/Users/Acer/Desktop/DribhyaDrishti/data/asl_alphabet"

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     dataset_path,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     dataset_path,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
# )

# normalization_layer = tf.keras.layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(26, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_ds, validation_data=val_ds, epochs=10)
# model.save("../models/static_model.h5")

import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Constants
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
EPOCHS = 30

# Paths
dataset_path = "C:/Users/Acer/Desktop/DribhyaDrishti/data/asl_alphabet"
model_save_path = "C:/Users/Acer/Desktop/DribhyaDrishti/models"

# Ensure model directory exists
os.makedirs(model_save_path, exist_ok=True)

# Load and preprocess dataset
def load_and_preprocess_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    print(f"Class names: {class_names}")
    return train_ds, val_ds, class_names

# Data augmentation
def create_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomFlip("horizontal"),
    ])

# Create model
def create_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = create_augmentation_layer()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

# Callbacks
def create_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(os.path.join(model_save_path, 'training_log.csv'))
    ]

# Plot history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(model_save_path, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    plt.show()

# Main function
def main():
    train_ds, val_ds, class_names = load_and_preprocess_data()
    model = create_model((*IMG_SIZE, 3), len(class_names))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=create_callbacks()
    )

    # Save final model (format inferred from .keras)
    final_model_path = os.path.join(model_save_path, 'final_model.keras')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training history
    plot_training_history(history)

    # Save class names
    class_names_path = os.path.join(model_save_path, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"Class names saved to {class_names_path}")

if __name__ == "__main__":
    main()
