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

def ResUNet(input_shape=(128, 128, 1)):
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
