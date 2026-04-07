# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 08:51:26 2026

@author: ai14
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# Train/Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Reshape images
X_train = X_train.reshape(X_train.shape[0], 128, 128, 3)
X_test  = X_test.reshape(X_test.shape[0], 128, 128, 3)

# Normalize (IMPORTANT for MobileNetV2)
X_train = X_train / 255.0
X_test  = X_test / 255.0

# ==========================================
# MobileNetV2 Base Model
# ==========================================
base_model = MobileNetV2(
    input_shape=(128, 128, 3),   # changed from 224 → 128
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# ==========================================
# Custom Top Layers (similar to your design)
# ==========================================
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

outputs = Dense(8, activation='softmax')(x)  # same number of classes

model = Model(inputs=base_model.input, outputs=outputs)

model.summary()

# ==========================================
# Compile
# ==========================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Callbacks
# ==========================================
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

# ==========================================
# Train
# ==========================================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
# #####################################
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# Base Model
# ==========================================
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# ==========================================
# Deeper Head (like CNN fully connected part)
# ==========================================
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Block 1
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Block 2
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Block 3
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Block 4 (extra depth)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Output
outputs = Dense(8, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.summary()