# ------------------------------
# 1. 🧩 Import Required Libraries
# ------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import flwr as fl
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose,
    LayerNormalization, Dense, Layer, Add, Reshape, concatenate
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

warnings.filterwarnings("ignore")

# ------------------------------
# 2. 📥 Load and Split Data
# ------------------------------
images = np.load('client_5_images.npy')
masks = np.load('client_5_masks.npy')
assert images.shape[0] == masks.shape[0], "Mismatch in number of samples."

X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.30, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True)

print(f"Train Images: {X_train.shape}, Masks: {y_train.shape}")
print(f"Val Images:   {X_val.shape}, Masks: {y_val.shape}")
print(f"Test Images:  {X_test.shape}, Masks: {y_test.shape}")

# ------------------------------
# 3. 🎯 Define Loss Functions
# ------------------------------
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

# ------------------------------
# 4. 🧠 Model Imports
# ------------------------------

#Doing the model imports as per their sequence
from models import get_model_1
model = get_model_1(input_shape=(128, 128, 1))
model.summary()

# ------------------------------
# 7. 🌐 Federated Client Setup
# ------------------------------
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.round = 0

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)

        checkpoint = ModelCheckpoint("best_model5.h5", monitor="val_loss", save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            batch_size=16,
                            epochs=500,
                            callbacks=[checkpoint, reduce_lr])

        pd.DataFrame(history.history).plot()
        plt.title(f"Client 5 - Round {self.round}")
        plt.xlabel("Epochs")
        plt.savefig(f"Loss_accu_image_fed_round_client5_{self.round}.png")
        self.round += 1

        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        print("Eval accuracy:", accuracy)
        return loss, len(X_test), {"accuracy": accuracy}

# Start Federated Client
fl.client.start_client(server_address="127.0.0.1:18080", client=CifarClient().to_client())

# ------------------------------
# 8. 📏 Calculate Evaluation Metrics
# ------------------------------
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

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 1.0

    return iou, dice, accuracy, recall, precision

# Predict and Compute Metrics
pred_masks = model.predict(X_test)
Y_pred_bin = (pred_masks > 0.5).astype("float32")

metrics_list = []
for i in range(y_test.shape[0]):
    iou, dice, acc, rec, prec = calculate_iou_and_dice(y_test[i], Y_pred_bin[i])
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

# ------------------------------
# 9. 💾 Save Predictions and Ground Truth
# ------------------------------
np.save('X_test5.npy', X_test)
np.save('Y_test5.npy', y_test)
np.save('Y_pred5.npy', Y_pred_bin)
