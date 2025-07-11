# ---------------------------------------------
# 📚 IMPORT LIBRARIES
# ---------------------------------------------
import flwr as fl                          # Federated Learning Framework
import numpy as np                         # Numerical computations
import pandas as pd                        # Data handling
import matplotlib.pyplot as plt            # Plotting
import tensorflow as tf                    # Deep Learning Framework
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, 
                                     LayerNormalization, Dense, Layer, Add, Reshape, concatenate)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")          # Suppress warnings
# Dice Loss: Measures overlap between predicted and true masks
def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.keras.backend.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# Focal Loss: Focuses on harder examples during training
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = alpha * (1 - p_t) ** gamma * bce
    return tf.keras.backend.mean(loss)

# Combined Loss: Weighted sum of Dice and Focal Loss
def combined_loss(y_true, y_pred):
    return 0.9 * dice_loss(y_true, y_pred) + 0.1 * focal_loss(y_true, y_pred)

def get_model(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)  # Input layer
    
    # Encoder: Convolution layers mimicking VGG
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # Conv Layer 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)       # Conv Layer 2
    pool1 = MaxPooling2D((2, 2))(x)                                     # Downsample

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)   # Conv Layer 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)       # Conv Layer 4
    pool2 = MaxPooling2D((2, 2))(x)                                     # Downsample

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)   # Conv Layer 5
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)       # Conv Layer 6
    pool3 = MaxPooling2D((2, 2))(x)                                     # Downsample

    # Output score and upsample to match input resolution
    score = Conv2D(1, (1, 1), activation='sigmoid')(pool3)              # Classifier layer
    score = Conv2DTranspose(1, kernel_size=8, strides=8, padding='same')(score)  # Upsample to original size

    model = Model(inputs, score)                                       # Define model
    model.compile(optimizer=AdamW(1e-4), loss=combined_loss, metrics=['accuracy'])  # Compile with optimizer and loss
    return model
