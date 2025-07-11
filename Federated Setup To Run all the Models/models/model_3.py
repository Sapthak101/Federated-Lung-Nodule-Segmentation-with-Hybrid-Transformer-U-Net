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
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, 
                                     LayerNormalization, Dense, Add, concatenate, 
                                     AveragePooling2D, UpSampling2D, BatchNormalization)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

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

def res_block(x, filters):
    shortcut = x  # Save shortcut for skip connection
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)  # First conv
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)  # Second conv
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, 1, padding='same')(shortcut)  # Match dimensions
    x = Add()([x, shortcut])                                 # Residual addition
    x = Activation('relu')(x)
    return x

def get_model(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = res_block(inputs, 64)                  # First residual block
    pool1 = MaxPooling2D()(conv1)                  # Downsample

    conv2 = res_block(pool1, 128)                  # Second residual block
    pool2 = MaxPooling2D()(conv2)                  # Downsample

    conv3 = res_block(pool2, 256)                  # Bottleneck

    # Decoder
    up4 = Conv2DTranspose(128, 2, strides=2, padding='same')(conv3)  # Upsample
    up4 = concatenate([up4, conv2])                                   # Skip connection
    up4 = res_block(up4, 128)                                         # Residual decode block

    up5 = Conv2DTranspose(64, 2, strides=2, padding='same')(up4)      # Upsample
    up5 = concatenate([up5, conv1])                                   # Skip connection
    up5 = res_block(up5, 64)                                          # Final decode block

    outputs = Conv2D(1, 1, activation='sigmoid')(up5)                 # Final mask prediction

    model = Model(inputs, outputs)
    model.compile(optimizer=AdamW(1e-4), loss=combined_loss, metrics=['accuracy'])
    return model
