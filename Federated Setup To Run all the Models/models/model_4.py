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

from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Activation

# ---------------------- CONV BLOCK ----------------------
def conv_block_3d(inputs, filters):
    x = Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# ---------------------- COMBINED LOSS (you can customize as needed) ----------------------
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice_loss = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)
    return bce + dice_loss

# ---------------------- 3D U-NET ARCHITECTURE ----------------------
def build_3d_unet_single_depth(input_shape=(128, 128, 1)):
    """
    Builds a 3D U-Net where input shape is (128, 128, 1) and is reshaped internally to (128, 128, 1, 1)
    Output shape is (128, 128, 1)
    """
    inputs = Input(input_shape)

    # Reshape to add depth dimension: (128, 128, 1) → (128, 128, 1, 1)
    x = Reshape((128, 128, 1, 1))(inputs)

    # Encoder
    c1 = conv_block_3d(x, 32)
    p1 = MaxPooling3D(pool_size=(2, 2, 1))(c1)

    c2 = conv_block_3d(p1, 64)
    p2 = MaxPooling3D(pool_size=(2, 2, 1))(c2)

    c3 = conv_block_3d(p2, 128)
    p3 = MaxPooling3D(pool_size=(2, 2, 1))(c3)

    c4 = conv_block_3d(p3, 256)
    d4 = Dropout(0.5)(c4)

    # Bottleneck
    c5 = conv_block_3d(d4, 512)
    d5 = Dropout(0.5)(c5)

    # Decoder
    u6 = Conv3DTranspose(256, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(d5)
    u6 = concatenate([u6, c4])
    c6 = conv_block_3d(u6, 256)

    u7 = Conv3DTranspose(128, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block_3d(u7, 128)

    u8 = Conv3DTranspose(64, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block_3d(u8, 64)

    u9 = Conv3DTranspose(32, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block_3d(u9, 32)

    # Final layer
    conv_final = Conv3D(1, kernel_size=(1, 1, 1), activation='sigmoid')(c9)  # Output shape: (128, 128, 1, 1)

    # Remove the depth dimension to get output shape (128, 128, 1)
    output = Reshape((128, 128, 1))(conv_final)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=combined_loss, metrics=['accuracy'])

    return model
