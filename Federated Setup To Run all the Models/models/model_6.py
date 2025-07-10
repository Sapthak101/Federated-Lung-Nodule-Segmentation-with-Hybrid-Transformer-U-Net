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

class TransformerBlock(Layer):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=dim)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            Dense(dim * mlp_ratio, activation='relu'),
            Dense(dim)
        ])
        self.dropout = Dropout(dropout)

    def call(self, x):
        x = self.norm1(x + self.dropout(self.attn(x, x)))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x

def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal")(shortcut)

    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = LayerNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = LayerNormalization()(x)

    x = Add()([shortcut, x])
    x = tf.keras.activations.relu(x)
    return x

def Final_BCDU_Transformer(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    conv1 = residual_block(inputs, 64)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = residual_block(pool1, 128)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = residual_block(pool2, 256)
    drop3 = Dropout(0.6)(conv3)
    pool3 = MaxPooling2D((2, 2))(drop3)

    trans = Reshape((16*16, 256))(pool3)
    trans = TransformerBlock(256)(trans)
    trans = TransformerBlock(256)(trans)
    trans = Reshape((16, 16, 256))(trans)

    up6 = Conv2DTranspose(128, 2, strides=2, padding='same')(trans)
    up6 = concatenate([conv3, up6])
    up6 = residual_block(up6, 128)

    up7 = Conv2DTranspose(64, 2, strides=2, padding='same')(up6)
    up7 = concatenate([conv2, up7])
    up7 = residual_block(up7, 64)

    up8 = Conv2DTranspose(64, 2, strides=2, padding='same')(up7)
    up8 = residual_block(up8, 64)

    output = Conv2D(1, 1, activation='sigmoid')(up8)

    model = Model(inputs, output)
    model.compile(optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
                  loss=combined_loss, metrics=['accuracy'])
    return model