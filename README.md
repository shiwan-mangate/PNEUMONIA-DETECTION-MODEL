#  Chest X-Ray Pneumonia Detection

A deep learning–based system that detects **pneumonia** from chest X-ray images using advanced convolutional neural networks — **ResNet** and **EfficientNet**.  
The project provides a complete training pipeline, model tuning framework, and an interactive **Streamlit** web interface for real-time predictions.

---

##  Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Model Architecture](#model-architecture)
- [Training & Hyperparameter Tuning](#training--hyperparameter-tuning)
- [Performance](#performance)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Future Work](#future-work)
- [References](#references)

---

##  Features
- Detects pneumonia vs. normal cases from X-ray images  
- Supports multiple model architectures:
  - **ResNet (18, 34, 50, 101, 152)**
  - **EfficientNet (B0–B7)**
- Configurable input channels (grayscale or RGB)  
- Integrated **Optuna**-based hyperparameter tuning  
- Generates **classification reports** and **confusion matrices**  
- Lightweight **Streamlit** UI for fast image-based predictions  

---

- All images are resized to **224×224**
- Converted to **grayscale**
- Dataset used: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

##  Data Preprocessing & Augmentation

### Training Augmentations
- Random horizontal flip  
- Random rotation (±15°)  
- Random affine transformations (±5%)  
- Random resized crop (scale: 90–110%)  
- Random sharpness adjustment  
- Grayscale conversion  
- Normalization `(mean=0.5, std=0.5)`

### Testing Transformations
- Resize to 224×224  
- Grayscale conversion  
- Normalization `(mean=0.5, std=0.5)`

---

##  Model Architecture

### 🔹 ResNet
- Pretrained variants: 18, 34, 50, 101, 152  
- Selectable number of trainable layers for fine-tuning  
- Dropout applied before the fully connected layer  
- Adjustable input channels

---

##  Training & Hyperparameter Tuning
- **Loss Function:** `CrossEntropyLoss`  
- **Optimizer:** `Adam`  
- **Batch Size:** 32  
- **Epochs:** 5 (customizable)  
- **Device:** GPU if available  

### Hyperparameters Tuned (via Optuna)
- Learning rate  
- Dropout rate  
- Number of trainable layers (ResNet) / blocks (EfficientNet)

---

## Performance

| Model         | Accuracy | Class 0 (Normal) | Class 1 (Pneumonia) | Macro F1 | Weighted F1 |
|---------------|-----------|-----------------|--------------------|----------|-------------|
| **ResNet**    | **94%**   | P:0.98 R:0.76 F1:0.86 | P:0.88 R:0.99 F1:0.93 | 0.90 | 0.90 |
| **EfficientNet** | 90%   | P:0.95 R:0.78 F1:0.86 | P:0.88 R:0.98 F1:0.93 | 0.89 | 0.90 |

 **Best accuracy achieved:** 94% using **ResNet**

---

## Future Work
Integration of Grad-CAM for visual explainability

Expansion to multi-class classification (e.g., COVID, TB, etc.)

Deploying the model as a cloud inference service (API-based)

Model ensembling for improved robustness
Kaggle Pneumonia Dataset

