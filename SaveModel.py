import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from imblearn.over_sampling import RandomOverSampler

sns.set_style("darkgrid")
warnings.filterwarnings("ignore")

# ==========================================
# Dataset Path
# ==========================================

dataset_path = "D:/Skin_disease/ISIC_2019/"

# ==========================================
# Load and Resize Images
# ==========================================

def load_and_resize_images(image_paths, target_size=(128,128)):

    images = []

    for path in image_paths:

        img = load_img(path, target_size=target_size)

        img = img_to_array(img) / 255.0

        images.append(img)

    return np.array(images)

# ==========================================
# Classes
# ==========================================

class_folders = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

all_images = []
all_labels = []

for class_name in class_folders:

    class_folder = os.path.join(dataset_path, class_name)

    image_files = os.listdir(class_folder)

    image_paths = [os.path.join(class_folder, img) for img in image_files]

    class_images = load_and_resize_images(image_paths)

    all_images.append(class_images)

    all_labels.append([class_name]*len(image_files))

X_all = np.concatenate(all_images, axis=0)
y_all = np.concatenate(all_labels, axis=0)

print("Images shape:", X_all.shape)
print("Labels shape:", y_all.shape)

# ==========================================
# Label Encoding
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y_all)

y_onehot = to_categorical(y_encoded)

# ==========================================
# Oversampling
# ==========================================

X_reshaped = X_all.reshape(X_all.shape[0], -1)

oversampler = RandomOverSampler(random_state=42)

X_resampled, y_resampled = oversampler.fit_resample(X_reshaped, y_onehot)

print("Class distribution:", Counter(np.argmax(y_resampled,axis=1)))

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# reshape back to images

X_train = X_train.reshape(X_train.shape[0],128,128,3)
X_test = X_test.reshape(X_test.shape[0],128,128,3)

# ==========================================
# ==========GlobalAveragePooling2D instead of Flatten (reduces overfitting)===
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Input
inputs = Input(shape=(128,128,3))

# Block 1
x = Conv2D(32,(3,3),padding='same',activation='relu',
           kernel_initializer='he_normal')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

# Block 2
x = Conv2D(64,(3,3),padding='same',activation='relu',
           kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

# Block 3
x = Conv2D(128,(3,3),padding='same',activation='relu',
           kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

# Block 4
x = Conv2D(256,(3,3),padding='same',activation='relu',
           kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

# Global Pooling
x = GlobalAveragePooling2D()(x)

# Dense Layers
x = Dense(256,activation='relu',kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)

x = Dense(128,activation='relu',kernel_regularizer=l2(0.001))(x)
x = Dropout(0.4)(x)

# Output
outputs = Dense(8,activation='softmax')(x)

model = Model(inputs,outputs)

model.summary()
# =============================================================================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
# ==============================================================================
# CNN Model (Functional API)
# ==========================================

inputs = Input(shape=(128,128,3))

# Block 1
x = Conv2D(32,(3,3),
           activation='relu',
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(0.01))(inputs)

x = MaxPooling2D()(x)
x = BatchNormalization()(x)

# Block 2
x = Conv2D(64,(3,3),
           activation='relu',
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(0.01))(x)

x = MaxPooling2D()(x)
x = BatchNormalization()(x)

# Block 3
x = Conv2D(128,(3,3),
           activation='relu',
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(0.01))(x)

x = MaxPooling2D()(x)
x = BatchNormalization()(x)

# Flatten
x = Flatten()(x)

# Dense layers
x = Dropout(0.5)(x)

x = Dense(256,
          activation='relu',
          kernel_initializer='he_normal',
          kernel_regularizer=l2(0.01))(x)

x = BatchNormalization()(x)

x = Dense(128,
          activation='relu',
          kernel_initializer='he_normal',
          kernel_regularizer=l2(0.01))(x)

x = BatchNormalization()(x)

x = Dense(64,
          activation='relu',
          kernel_initializer='he_normal',
          kernel_regularizer=l2(0.01))(x)

x = BatchNormalization()(x)

# Output
outputs = Dense(8, activation='softmax')(x)

# Model
model = Model(inputs=inputs, outputs=outputs)

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
restore_best_weights=True)

]

# ==========================================
# Train Model
# ==========================================

history = model.fit(

X_train,
y_train,

validation_data=(X_test,y_test),

epochs=100,

batch_size=32,

callbacks=callbacks

)

# ==========================================
# Save Model
# ==========================================

model.save("D:/Skin_disease/final_code_1/website/models/CNN_1032026.h5")
print("Model saved successfully")
# =============================================================================
import os, warnings, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.layers import TimeDistributed, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# --- Dataset path ---
dataset_path = "D:/Skin_disease/ISIC_2019/"

# --- Load and resize images ---
def load_and_resize_images(image_paths, target_size=(128,128)):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=target_size)
        img = img_to_array(img)/255.0
        images.append(img)
    return np.array(images)

# --- Classes ---
class_folders = ['MEL','VASC','SCC','DF','NV','BKL','BCC','AK']
all_images, all_labels = [], []

for class_name in class_folders:
    folder = os.path.join(dataset_path, class_name)
    files = os.listdir(folder)
    paths = [os.path.join(folder,f) for f in files]
    class_images = load_and_resize_images(paths)
    all_images.append(class_images)
    all_labels.append([class_name]*len(files))

# --- Combine ---
X_all = np.concatenate(all_images,axis=0)
y_all = np.concatenate(all_labels,axis=0)

# --- Encode labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y_all)
y_onehot = to_categorical(y_encoded)

# --- Oversampling ---
X_flat = X_all.reshape(X_all.shape[0],-1)
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_flat,y_onehot)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2,random_state=42)
X_train = X_train.reshape(X_train.shape[0],1,128,128,3)
X_test = X_test.reshape(X_test.shape[0],1,128,128,3)

# --- Hybrid CNN-LSTM (Functional API) ---
inputs = Input(shape=(1,128,128,3))

x = TimeDistributed(Conv2D(32,(3,3),activation='relu',padding='same'))(inputs)
x = TimeDistributed(MaxPooling2D())(x)
x = TimeDistributed(BatchNormalization())(x)

x = TimeDistributed(Conv2D(64,(3,3),activation='relu',padding='same'))(x)
x = TimeDistributed(MaxPooling2D())(x)
x = TimeDistributed(BatchNormalization())(x)

x = TimeDistributed(Conv2D(128,(3,3),activation='relu',padding='same'))(x)
x = TimeDistributed(MaxPooling2D())(x)
x = TimeDistributed(BatchNormalization())(x)

x = TimeDistributed(Flatten())(x)
x = LSTM(128,return_sequences=False)(x)

x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128,activation='relu')(x)

outputs = Dense(8,activation='softmax')(x)
model = Model(inputs,outputs)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',factor=0.3,patience=4,min_lr=1e-6)
]

# --- Train ---
history = model.fit(X_train,y_train,
                    validation_data=(X_test,y_test),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks)

# --- Save model ---
model.save("Hybrid_CNN_LSTM_ISIC2019.h5")
print("Model saved successfully!")

# --- Evaluation ---
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

print(classification_report(y_true,y_pred_classes))

cm = confusion_matrix(y_true,y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
# =============================================================================
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.layers import TimeDistributed, LSTM
from tensorflow.keras.models import Model

# Input
inputs = Input(shape=(1,128,128,3), name="input_image")

# --- CNN Block 1 ---
x = TimeDistributed(Conv2D(32,(3,3),activation='relu',padding='same',name='conv2d_block1'), name='td_conv1')(inputs)
x = TimeDistributed(MaxPooling2D(name='maxpool_block1'), name='td_maxpool1')(x)
x = TimeDistributed(BatchNormalization(name='bn_block1'), name='td_bn1')(x)

# --- CNN Block 2 ---
x = TimeDistributed(Conv2D(64,(3,3),activation='relu',padding='same',name='conv2d_block2'), name='td_conv2')(x)
x = TimeDistributed(MaxPooling2D(name='maxpool_block2'), name='td_maxpool2')(x)
x = TimeDistributed(BatchNormalization(name='bn_block2'), name='td_bn2')(x)

# --- CNN Block 3 ---
x = TimeDistributed(Conv2D(128,(3,3),activation='relu',padding='same',name='conv2d_block3'), name='td_conv3')(x)
x = TimeDistributed(MaxPooling2D(name='maxpool_block3'), name='td_maxpool3')(x)
x = TimeDistributed(BatchNormalization(name='bn_block3'), name='td_bn3')(x)

# --- Flatten CNN features ---
x = TimeDistributed(Flatten(name='flatten_cnn'), name='td_flatten')(x)

# --- LSTM Layer ---
x = LSTM(128, return_sequences=False, name='lstm_layer')(x)

# --- Dense Layers ---
x = Dense(256, activation='relu', name='dense_256')(x)
x = Dropout(0.5, name='dropout_256')(x)
x = Dense(128, activation='relu', name='dense_128')(x)

# --- Output Layer ---
outputs = Dense(8, activation='softmax', name='output')(x)

# --- Create Model ---
model = Model(inputs, outputs, name="Hybrid_CNN_LSTM")

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# ------------------------Grad-CAM
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===============================
# Set the last CNN layer name
# ===============================
last_conv_layer_name = "conv2d_block3"

# ===============================
# Build gradient model
# ===============================
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

# ===============================
# Grad-CAM Function
# ===============================
def gradcam(image):
    """
    image: single image of shape (128,128,3)
    returns: heatmap of shape (128,128)
    """
    image = np.expand_dims(image, axis=0)  # add batch
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap,0)
    heatmap = heatmap / np.max(heatmap)
    return heatmap.numpy()

# ===============================
# Grad-CAM++ Function
# ===============================
def gradcam_plus(image):
    """
    Grad-CAM++ heatmap
    """
    image = np.expand_dims(image, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    grads_power = grads**2
    grads_cube = grads**3
    sum_activations = tf.reduce_sum(conv_outputs, axis=(0,1,2))
    alpha = grads_power / (2*grads_power + grads_cube*sum_activations + 1e-8)
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap = heatmap / np.max(heatmap)
    return heatmap.numpy()

# ===============================
# Overlay Heatmap Function
# ===============================
def show_gradcam(image):
    heatmap1 = gradcam(image)
    heatmap2 = gradcam_plus(image)

    heatmap1 = cv2.resize(heatmap1, (128,128))
    heatmap2 = cv2.resize(heatmap2, (128,128))

    heatmap1 = cv2.applyColorMap(np.uint8(255*heatmap1), cv2.COLORMAP_JET)
    heatmap2 = cv2.applyColorMap(np.uint8(255*heatmap2), cv2.COLORMAP_JET)

    img = (image*255).astype(np.uint8)

    superimposed1 = cv2.addWeighted(img,0.6,heatmap1,0.4,0)
    superimposed2 = cv2.addWeighted(img,0.6,heatmap2,0.4,0)

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(superimposed1)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(superimposed2)
    plt.title("Grad-CAM++")
    plt.axis("off")

    plt.show()

# ===============================
# Example Usage
# ===============================
sample_image = X_test[0][0]  # remove sequence dimension
show_gradcam(sample_image)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@segmetation
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# ==========================================
# Dataset Path
# ==========================================
dataset_path = "D:/Skin_disease/ISIC_2019/"

# ==========================================
# Load and Resize Images
# ==========================================
def load_and_resize_images(image_paths, target_size=(128,128)):

    images = []

    for path in image_paths:
        img = load_img(path, target_size=target_size)
        img = img_to_array(img) / 255.0
        images.append(img)

    return np.array(images)

# ==========================================
# Classes
# ==========================================
class_folders = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

all_images = []
all_labels = []

for class_name in class_folders:

    class_folder = os.path.join(dataset_path, class_name)
    image_files = os.listdir(class_folder)
    image_paths = [os.path.join(class_folder, img) for img in image_files]

    class_images = load_and_resize_images(image_paths)

    all_images.append(class_images)
    all_labels.append([class_name]*len(image_files))

X_all = np.concatenate(all_images, axis=0)
y_all = np.concatenate(all_labels, axis=0)

print("Images shape:", X_all.shape)
print("Labels shape:", y_all.shape)

# ==========================================
# Label Encoding (original - not used later)
# ==========================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_all)
y_onehot = to_categorical(y_encoded)

# ==========================================
# Convert to Skin vs Non-Skin
# ==========================================
y_skin = np.ones((X_all.shape[0], 1))  # ISIC images = skin

# Create artificial non-skin data
num_samples = len(X_all)
X_non_skin = np.random.rand(num_samples, 128, 128, 3)
y_non_skin = np.zeros((num_samples, 1))

# Combine
X_binary = np.concatenate([X_all, X_non_skin], axis=0)
y_binary = np.concatenate([y_skin, y_non_skin], axis=0)

print("Binary dataset:", X_binary.shape, y_binary.shape)

# ==========================================
# Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Build Skin Detector Model
# ==========================================
inputs = Input(shape=(128,128,3))

x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = MaxPooling2D()(x)

x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D()(x)

x = Conv2D(128, 3, activation='relu', padding='same')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# Train Model
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# ==========================================
# Save Model
# ==========================================
model.save("D:/Skin_disease/final_code_1/website/models/skin_detector.h5")
print("Skin detector model saved successfully!")

# ==========================================
# Skin Percentage Function (Patch-based)
# ==========================================
def calculate_skin_percentage(model, image, patch_size=32):

    h, w, _ = image.shape
    skin_count = 0
    total_patches = 0

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):

            patch = image[y:y+patch_size, x:x+patch_size]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            patch = np.expand_dims(patch, axis=0)

            pred = model.predict(patch, verbose=0)

            if pred > 0.5:
                skin_count += 1

            total_patches += 1

    percentage = (skin_count / total_patches) * 100
    return percentage

# ==========================================
# Example Test
# ==========================================
test_img = X_test[0]
percentage = calculate_skin_percentage(model, test_img)

print("Skin Percentage:", percentage)