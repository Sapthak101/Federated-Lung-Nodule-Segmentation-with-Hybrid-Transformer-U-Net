# ---------------------- CONV BLOCK ----------------------
def conv_block_3d(inputs, filters):
    x = Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# ---------------------- 3D U-NET ARCHITECTURE ----------------------
def build_3d_unet_single_depth(input_shape=(128, 128, 1, 1)):
    """3D U-Net for (128, 128, 1, 1) input volume and (128, 128, 1) mask"""
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block_3d(inputs, 32)
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

    # Final 3D Conv output: shape = (128, 128, 1, 1)
    conv_final = Conv3D(1, kernel_size=(1, 1, 1), activation='sigmoid')(c9)

    # Remove the depth dimension: reshape (128, 128, 1, 1) → (128, 128, 1)
    output = Reshape((128, 128, 1))(conv_final)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=combined_loss, metrics=['accuracy'])
    return model
