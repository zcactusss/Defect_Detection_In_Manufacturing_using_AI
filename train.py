import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =====================
# DATA PATH
# =====================
train_path = 'dataset/train'

# =====================
# DATA AUGMENTATION (NO validation_split)
# =====================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

# =====================
# LOAD DATASET
# =====================
train_gen = datagen.flow_from_directory(
    train_path,
    target_size=(300, 300),
    batch_size=2,
    class_mode='binary'
)

print("Class Indices:", train_gen.class_indices)

# =====================
# SIMPLE CNN MODEL
# =====================
model = models.Sequential([
    layers.Input(shape=(300, 300, 3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# =====================
# COMPILE MODEL
# =====================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================
# TRAIN MODEL
# =====================
history = model.fit(
    train_gen,
    epochs=10
)

# =====================
# SAVE MODEL
# =====================
model.save("defect_model.keras")
print("Model Saved Successfully!")

# =====================
# PLOT (SAFE CHECK)
# =====================
if history.history.get('accuracy'):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Loss')
    plt.title("Loss")
    plt.legend()

    plt.show()
