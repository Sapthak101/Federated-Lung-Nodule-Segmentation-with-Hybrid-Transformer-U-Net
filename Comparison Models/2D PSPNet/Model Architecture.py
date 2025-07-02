def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]  # Start with original feature map
    h, w = input_tensor.shape[1:3]

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(h // bin_size, w // bin_size))(input_tensor)  # Pool to bin size
        x = Conv2D(64, (1, 1), padding='same')(x)                                     # Reduce channels
        x = UpSampling2D(size=(h // bin_size, w // bin_size), interpolation='bilinear')(x)  # Resize back
        concat_list.append(x)                                                        # Append to list

    return concatenate(concat_list)  # Concatenate all pooled features

def PSPNet(input_shape=(128, 128, 1)):
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
