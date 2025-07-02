
# 🧠 Federated Lung Nodule Segmentation with Hybrid Transformer-U-Net

> A privacy-preserving federated learning pipeline for solid lung nodule segmentation using a Hybrid Transformer-U-Net architecture with Dice-Focal loss.

---

## 📘 Overview

This project presents a federated learning framework designed to perform **solid lung nodule segmentation** on thoracic CT scans using a **Hybrid Transformer-U-Net architecture**. The model integrates both convolutional and attention-based modules for robust local and global feature learning while maintaining **data privacy** through federated training using the [Flower](https://flower.dev) framework.

### ✨ Key Contributions

- 🚀 A **Hybrid Transformer-U-Net** for effective local-global contextual learning.
- 🎯 A **custom Dice-Focal Loss** tailored to segment small and imbalanced lung nodules.
- 🌐 A **federated setup** with 5 clients and a server, emulated on a single system.
- 📊 Extensive comparison with state-of-the-art architectures under **federated settings**.

---

## 📂 Project Structure

```
├── Comparison Models/
│   ├── FCN-8/Model Architecture.py
│   ├── 2D PSPNet/Model Architecture.py
│   ├── 2D ResUNet/Model Architecture.py
│   ├── 3D U-Net/Model Architecture.py
│   ├── MSS U-Net/Model Architecture.py
│
├── Proposed Architecture/
│   ├── Client Scripts/client1.py
│   ├── Client Scripts/client2.py
│   ├── Client Scripts/client3.py
│   ├── Client Scripts/client4.py
│   ├── Client Scripts/client5.py
│   └── Server Script/server.py
│
├── Preprocessing Data/
│   └── Preprocessing_Data.ipynb
```

---

## 🖼️ Dataset

### 📎 Original Dataset (LUNA16 - LIDC-IDRI)
- Hugging Face: [LUNA16 Segmentation Data](https://huggingface.co/datasets/H-Huang/LUNA16_segmentation_data)
- Clone:
  ```bash
  git clone https://huggingface.co/datasets/H-Huang/LUNA16_segmentation_data
  ```

### 🔧 Preprocessed Dataset
Client-wise preprocessed dataset split for federated learning (in `.npy` format):

📥 [Google Drive - Preprocessed Data (5 Clients)](https://drive.google.com/drive/folders/1MveBBH-sH7I4FRWXiGZ_fnj0-9k7kyxw)

Each directory contains:
```
client_x_images.npy
client_x_masks.npy
```

---

## ⚙️ Preprocessing Description

The following preprocessing steps were applied:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
   Enhances visibility of low-contrast nodules.  
   - Clip limit: `2.0`  
   - Tile grid size: `8×8`

2. **Min-Max Normalization**  
   Scales pixel values to [0, 1] range for better convergence.

📁 Code: `Preprocessing Data/Preprocessing_Data.ipynb`

---

## 🧠 Proposed Model: `Final_BCDU_Transformer()`

### 📌 Description
A **Federated Hybrid Transformer Residual U-Net** model integrating:

- ✅ **Residual U-Net** for robust local feature extraction.
- ✅ **Transformer Blocks** for long-range context modeling.
- ✅ **Dice-Focal Loss** to handle extreme class imbalance (solid nodule vs background).

### 🎯 Why It Excels

- Strong generalization across diverse client data.
- Robust on **small**, **low-contrast**, and **heterogeneous** nodules.
- Efficient: **Dice Score** = `0.800` with only **7–8M** parameters.
- Designed for **federated deployment in clinical setups**.

---

## 🔬 Comparison Architectures

| Function                   | Full Model Name                                                                 | Description                                              |
| -------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `fcn8_model()`             | **Fully Convolutional Network (FCN-8s)**                                        | Early semantic segmentation; weak context capture        |
| `pspnet_2d_model()`        | **2D Pyramid Scene Parsing Network (2D PSPNet)**                                | Global context via pyramid pooling                       |
| `resunet_2d_model()`       | **2D Residual U-Net**                                                           | Residual-enhanced encoder-decoder                        |
| `unet_3d_model()`          | **3D U-Net**                                                                    | Captures volumetric features                             |
| `mssunet_2d_model()`       | **Multi-Scale Supervised U-Net (MSS U-Net)**                                    | Multi-resolution supervision for refinement              |
| `Final_BCDU_Transformer()` | **Federated Hybrid Transformer Residual U-Net with Dice-Focal Loss (Proposed)** | Transformer-enhanced U-Net under federated setup         |

📁 Code: `Comparison Models/`

🛠️ Replace the model in client scripts to test comparisons.

---

## 🖥️ How to Run (Federated Setup)

> Make sure all `client*.py` and `server.py` files are in the same directory.

### 1. Launch Server

```bash
python server.py
```

### 2. Launch Clients (in separate terminals)

```bash
python client1.py
python client2.py
python client3.py
python client4.py
python client5.py
```

---

## 🧪 Results Summary

| Model                               | Dice Score | Parameters |
| ----------------------------------- | ---------- | ---------- |
| FCN-8s                              | 0.310      | ~12M       |
| 2D PSPNet                           | 0.580      | ~22M       |
| 2D Res U-Net                        | 0.715      | ~15M       |
| 3D U-Net                            | 0.680      | ~30–35M    |
| MSS U-Net                           | 0.670      | ~18M       |
| **Ours (Hybrid Transformer-U-Net)** | **0.800**  | **7–8M**   |

---

## 🧾 Additional Comparative Studies

| Reference | Model | Dice Score | Parameters |
| --------- | ----- | ---------- | ---------- |
| Jain et al. (2021) | Metaheuristic-based GAN (SSSOA-GAN) | 0.7986 | ~32–38M |
| Yu et al. (2021)   | 3D Residual U-Net                    | 0.80 (for nodules >10mm) | ~30–35M |
| **Ours**           | Federated Hybrid Transformer U-Net   | **0.800** | **7–8M** |

---

## 🔐 Why Federated Learning?

- ✅ **GDPR/HIPAA compliant** – no data leaves local institutions.
- ✅ **Edge-deployable** – trains on-site at multiple hospitals.
- ✅ **Privacy-preserving** – only weights are shared.
- ✅ **Robust performance** – trains on diverse and distributed datasets.

---

## 💾 Best Model Checkpoints

📥 [Google Drive - Final Checkpoints](https://drive.google.com/drive/folders/1Pxd7hlU4ZAcKaEJnwqQyi2dEEKEcbQnl)

---

## 🧑‍💻 Contact

For research collaboration, code issues, or academic queries:

**📧 Email**: sapthakmohajon6@gmail.com

---
