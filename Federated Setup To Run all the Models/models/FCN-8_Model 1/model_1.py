def FCN8(input_shape=(128, 128, 1)):
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
