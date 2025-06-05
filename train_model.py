import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# KONFIGURASI
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10
DATASET_DIR = "dataset"

# AUGMENTATION DAN LOAD DATA


datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    "dataset/Train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    "dataset/Val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# TRANSFER LEARNING: EfficientNetB7
base_model = EfficientNetB7(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Bekukan base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# TRAINING
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# SIMPAN MODEL
model.save("model_grape_leaf_effnetb7.h5")
print("Model berhasil disimpan ke model_grape_leaf_effnetb7.h5")
