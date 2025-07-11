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

from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose,
                                     concatenate, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from tensorflow_addons.optimizers import AdamW

def get_model(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D()(conv2)

    # Bottleneck
    bottleneck = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    bottleneck = Dropout(0.3)(bottleneck)

    # Decoder
    up3 = Conv2DTranspose(128, 2, strides=2, padding='same')(bottleneck)
    up3 = concatenate([up3, conv2])
    up3 = Conv2D(128, 3, activation='relu', padding='same')(up3)

    # Auxiliary output (from 64x64) → upsample to 128x128
    aux1 = Conv2D(1, 1, activation='sigmoid')(up3)
    aux1 = UpSampling2D(size=(2, 2), interpolation='bilinear', name="aux1")(aux1)

    up4 = Conv2DTranspose(64, 2, strides=2, padding='same')(up3)
    up4 = concatenate([up4, conv1])
    up4 = Conv2D(64, 3, activation='relu', padding='same')(up4)

    final_output = Conv2D(1, 1, activation='sigmoid', name="main")(up4)

    model = Model(inputs, [final_output, aux1])
    model.compile(optimizer=AdamW(1e-4),
                  loss={"main": combined_loss, "aux1": combined_loss},
                  loss_weights={"main": 1.0, "aux1": 0.4},
                  metrics=["accuracy"])
    return model
