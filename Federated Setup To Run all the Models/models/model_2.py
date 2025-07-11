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
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, 
                                     LayerNormalization, Dense, Add, concatenate, 
                                     AveragePooling2D, UpSampling2D)

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

def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]  # Include the original tensor
    h, w = tf.shape(input_tensor)[1], tf.shape(input_tensor)[2]  # Dynamic height and width

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(input_tensor.shape[1] // bin_size, 
                                        input_tensor.shape[2] // bin_size))(input_tensor)
        x = Conv2D(64, (1, 1), padding='same')(x)
        x = tf.image.resize(x, size=(input_tensor.shape[1], input_tensor.shape[2]), method='bilinear')  # Resize to match input_tensor
        concat_list.append(x)

    return concatenate(concat_list, axis=-1)


def get_model(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)

    # Shallow encoder
    x = Conv2D(64, 3, padding='same', activation='relu')(inputs)       # First conv
    x = MaxPooling2D()(x)                                              # Downsample
    x = Conv2D(128, 3, padding='same', activation='relu')(x)           # Second conv
    x = MaxPooling2D()(x)                                              # Downsample

    # PSP module
    x = pyramid_pooling_block(x, bin_sizes=[1, 2, 3, 6])               # Multi-scale context
    x = Conv2D(128, 1, padding='same', activation='relu')(x)           # Reduce channels
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)         # Restore spatial size
    x = Conv2D(1, 1, activation='sigmoid')(x)                          # Output mask

    model = Model(inputs, x)
    model.compile(optimizer=AdamW(1e-4), loss=combined_loss, metrics=['accuracy'])
    return model

model=get_model(input_shape=(128, 128, 1))
print(model.summary())