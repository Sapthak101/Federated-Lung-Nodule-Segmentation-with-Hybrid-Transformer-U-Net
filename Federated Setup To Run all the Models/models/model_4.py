import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, Conv3DTranspose,
                                     BatchNormalization, Activation, Dropout,
                                     concatenate, Reshape)
from tensorflow.keras.optimizers import Adam


# ---------------------- 🔧 Loss Functions ----------------------
def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = alpha * (1 - p_t) ** gamma * bce
    return tf.reduce_mean(loss)

def combined_loss(y_true, y_pred):
    return 0.9 * dice_loss(y_true, y_pred) + 0.1 * focal_loss(y_true, y_pred)


# ---------------------- 🔧 Conv Block ----------------------
def conv_block_3d(inputs, filters):
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# ---------------------- ✅ 3D U-Net Model ----------------------
def get_model(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)
    x = Reshape((128, 128, 1, 1))(inputs)  # (H, W, D=1, C=1)

    # Encoder
    c1 = conv_block_3d(x, 32)
    p1 = MaxPooling3D(pool_size=(2, 2, 1))(c1)  # 64x64

    c2 = conv_block_3d(p1, 64)
    p2 = MaxPooling3D(pool_size=(2, 2, 1))(c2)  # 32x32

    c3 = conv_block_3d(p2, 128)
    p3 = MaxPooling3D(pool_size=(2, 2, 1))(c3)  # 16x16

    c4 = conv_block_3d(p3, 256)
    p4 = MaxPooling3D(pool_size=(2, 2, 1))(c4)  # 8x8

    # Bottleneck
    c5 = conv_block_3d(p4, 512)

    # Decoder
    u6 = Conv3DTranspose(256, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(c5)
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

    outputs_5d = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)  # Shape: (128, 128, 1, 1)
    outputs = Reshape((128, 128, 1))(outputs_5d)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss=combined_loss, metrics=['accuracy'])
    return model
model = get_model(input_shape=(128, 128, 1))
print(model.summary())