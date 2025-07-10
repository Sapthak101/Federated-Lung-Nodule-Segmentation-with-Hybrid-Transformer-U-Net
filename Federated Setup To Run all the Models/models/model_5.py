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

def MSS_UNet(input_shape=(128, 128, 1)):
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

    # Decoder with auxiliary output
    up3 = Conv2DTranspose(128, 2, strides=2, padding='same')(bottleneck)
    up3 = concatenate([up3, conv2])
    up3 = Conv2D(128, 3, activation='relu', padding='same')(up3)
    aux1 = Conv2D(1, 1, activation='sigmoid', name="aux1")(up3)  # Auxiliary loss from intermediate decoder

    up4 = Conv2DTranspose(64, 2, strides=2, padding='same')(up3)
    up4 = concatenate([up4, conv1])
    up4 = Conv2D(64, 3, activation='relu', padding='same')(up4)

    final_output = Conv2D(1, 1, activation='sigmoid', name="main")(up4)

    model = Model(inputs, [final_output, aux1])
    model.compile(optimizer=AdamW(1e-4),
                  loss={"main": combined_loss, "aux1": combined_loss},
                  loss_weights={"main": 1.0, "aux1": 0.4},  # Deep supervision loss weighting
                  metrics=["accuracy"])
    return model
