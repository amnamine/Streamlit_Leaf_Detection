# 🍃 Tomato Leaf Disease Classifier

An interactive web application built with **Streamlit** to detect and classify diseases in tomato leaves. The project leverages three powerful deep learning architectures—**DenseNet121**, **EfficientNetB3**, and **ResNet50**—to provide accurate diagnoses.

## 🚀 Features

- **Multi-Model Support**: Choose between three different architectures (PyTorch & TensorFlow).
- **Interactive UI**: User-friendly interface for uploading images and viewing results.
- **Top-5 Analysis**: Visualizes the top 5 predicted classes with confidence scores.
- **Robust Loading**: Includes a compatibility layer for loading Keras models across different versions.

## 🏗️ Supported Architectures

1.  **DenseNet121 (PyTorch)**: Excellent feature reuse through dense connections. Fast and lightweight.
2.  **EfficientNetB3 (TensorFlow)**: Optimized compound scaling for state-of-the-art accuracy-to-parameter ratio.
3.  **ResNet50 (PyTorch)**: Deep residual network with a custom classification head and aggressive data augmentation.

## 📂 Project Structure

- `streamlit_app.py`: The main application script.
- `leaf_densenet.ipynb`: Training and evaluation for the DenseNet121 model.
- `leaf_efficientnet.ipynb`: Training and evaluation for the EfficientNetB3 model.
- `leaf_resnet50.ipynb`: Training and evaluation for the ResNet50 model.
- `*.pth` / `*.h5`: Pre-trained model weights (ensure these are in the root directory).

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd leaf_interfacee
```

### 2. Install dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run streamlit_app.py
```

## 📋 Requirements

- Python 3.8+
- Streamlit
- PyTorch & Torchvision
- TensorFlow 2.20.0
- NumPy, Pillow, Matplotlib, Seaborn, Pandas, Scikit-learn

## 🧬 Classes Detected

The model can identify the following 10 categories:
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Miner
- Tomato Mosaic Virus
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Healthy

---
*Developed for Tomato Leaf Disease Detection Research.*
