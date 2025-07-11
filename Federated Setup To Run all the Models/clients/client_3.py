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
import os

# ---------------------------------------------
# 📁 LOAD CLIENT DATA AND SPLIT
# ---------------------------------------------
# Get path of the current script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data using full path
images = np.load(os.path.join(CURRENT_DIR, 'client_3_images.npy'))
masks = np.load(os.path.join(CURRENT_DIR, 'client_3_masks.npy'))

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
# 🧠 Model Imports
# ---------------------------------------------
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#Doing the model imports as per their sequence
from models import get_model_6
model = get_model_6(input_shape=(128, 128, 1))
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
            epochs=5,
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
