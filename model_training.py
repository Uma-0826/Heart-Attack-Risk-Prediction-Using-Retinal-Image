import os

# 🔹 Hide oneDNN log
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 🔹 Suppress TensorFlow INFO and WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset path
train_dir = r"C:\Heart\dataset\train"
val_dir   = r"C:\Heart\dataset\val"
# Load dataset using new API
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary"
)

# Normalize images (0–255 → 0–1)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# Optimize performance
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)



# CNN Model
model = Sequential([
    Input(shape=(224,224,3)),
    Conv2D(32, (3,3), activation='relu'), 
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save model
model.save("heart_attack_risk_model.keras")

# Plot accuracy
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend()
plt.show()