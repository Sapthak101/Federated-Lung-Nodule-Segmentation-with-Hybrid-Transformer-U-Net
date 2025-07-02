# ---------------------------------------------
# 🧩 LIBRARY IMPORTS (cleaned and grouped)
# ---------------------------------------------
import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, 
    LayerNormalization, Dense, Layer, Add, Reshape, concatenate
)

warnings.filterwarnings("ignore")

# ---------------------------------------------
# 📁 LOAD CLIENT 3 DATA & SPLIT
# ---------------------------------------------
images = np.load('client_3_images.npy')
masks = np.load('client_3_masks.npy')
assert images.shape[0] == masks.shape[0], "Mismatch in number of samples."

X_train, X_temp, y_train, y_temp = train_test_split(
    images, masks, test_size=0.30, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

print(f"Train Images: {X_train.shape}, Masks: {y_train.shape}")
print(f"Val Images:   {X_val.shape}, Masks: {y_val.shape}")
print(f"Test Images:  {X_test.shape}, Masks: {y_test.shape}")

# ---------------------------------------------
# 🎯 LOSS FUNCTIONS
# ---------------------------------------------
def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.keras.backend.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = alpha * (1 - p_t) ** gamma * bce
    return tf.keras.backend.mean(loss)

def combined_loss(y_true, y_pred):
    return 0.9 * dice_loss(y_true, y_pred) + 0.1 * focal_loss(y_true, y_pred)

# ---------------------------------------------
# 🧠 TRANSFORMER BLOCK
# ---------------------------------------------
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

# ---------------------------------------------
# 🔁 RESIDUAL BLOCK
# ---------------------------------------------
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

# ---------------------------------------------
# 🏗️ MODEL DEFINITION: BCDU-Transformer
# ---------------------------------------------
def Final_BCDU_Transformer(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = residual_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_block(pool2, 256)
    drop3 = Dropout(0.6)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    # Transformer Bottleneck
    trans = Reshape((16*16, 256))(pool3)
    trans = TransformerBlock(dim=256)(trans)
    trans = TransformerBlock(dim=256)(trans)
    trans = Reshape((16, 16, 256))(trans)

    # Decoder
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(trans)
    up6 = concatenate([conv3, up6], axis=3)
    up6 = residual_block(up6, 128)

    up7 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(up6)
    up7 = concatenate([conv2, up7], axis=3)
    up7 = residual_block(up7, 64)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(up7)
    up8 = residual_block(up8, 64)

    conv8 = Conv2D(1, 1, activation='sigmoid', padding='same')(up8)
    model = Model(inputs, conv8)

    model.compile(optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
                  loss=combined_loss,
                  metrics=['accuracy'])
    return model

# Build model
model = Final_BCDU_Transformer()
model.summary()

# ---------------------------------------------
# 🌐 FLOWER CLIENT 3 SETUP
# ---------------------------------------------
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.index1 = 0

    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)

        checkpoint = ModelCheckpoint("best_model3.h5", monitor="val_loss", save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=16,
            epochs=500,
            callbacks=[checkpoint, reduce_lr]
        )

        pd.DataFrame(history.history).plot()
        plt.xlabel('Epochs')
        plt.title("Client 3 Training Metrics")
        plt.savefig(f"Loss_accu_image_fed_round_client3{self.index1}.png")
        self.index1 += 1

        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        print("Eval accuracy : ", accuracy)
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(server_address="127.0.0.1:18080", client=CifarClient().to_client())

# ---------------------------------------------
# 📏 METRIC EVALUATION ON TEST DATA
# ---------------------------------------------
def calculate_iou_and_dice(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    iou = intersection / union if union != 0 else 1.0
    dice = (2. * intersection) / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) != 0 else 1.0

    tp = intersection
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 1.0

    return iou, dice, accuracy, recall, precision

# Predict on test set
pred_masks = model.predict(X_test)
Y_pred_bin = (pred_masks > 0.5).astype("float32")

# Evaluate per-sample metrics
metrics_list = []
for i in range(y_test.shape[0]):
    gt = y_test[i]
    pred = Y_pred_bin[i]
    iou, dice, acc, rec, prec = calculate_iou_and_dice(gt, pred)
    metrics_list.append({
        'Image': i,
        'IoU': iou,
        'Dice': dice,
        'Accuracy': acc,
        'Recall': rec,
        'Precision': prec
    })

df = pd.DataFrame(metrics_list)
print(df)

# ---------------------------------------------
# 💾 SAVE RESULTS
# ---------------------------------------------
np.save('X_test3.npy', X_test)
np.save('Y_test3.npy', y_test)
np.save('Y_pred3.npy', Y_pred_bin)
