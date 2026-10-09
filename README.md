<div align="center">

# 🫁 PneumoAI

**Deep Learning Web Application for Pneumonia Detection from Chest X-Ray Scans**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Overview

**PneumoAI** is an end-to-end deep learning web application that leverages **Convolutional Neural Networks (CNNs)** and **transfer learning** with **VGG16** to detect pneumonia from chest X-ray images.

The application provides a modern, responsive web interface built with **Flask**, **Bootstrap 5**, and a custom **CSS design system**, enabling users to upload radiographic scans and receive instantaneous AI-powered diagnostic predictions with confidence scoring.

> 🩺 **Primary Goal**: Demonstrate real-world implementation of medical image classification pipelines from data preprocessing to web deployment.

---

## ✨ Key Features

- **⚡ Instant Classification**: Analyzes chest X-ray scans in under 3 seconds.
- **🎯 High Accuracy**: Achieves **88.46% test accuracy** and **0.9468 AUC score** on hold-out evaluation data.
- **🎨 Glassmorphism & Modern UI**: Built with responsive layouts, smooth micro-interactions, and visual status indicators.
- **🔒 Privacy First**: Performs in-memory image processing without persisting uploaded scans to disk.
- **📱 Drag-and-Drop Uploads**: Supports interactive file selection with instant client-side preview.
- **🛡️ Robust Error Handling**: Automatic file validation for `.jpg`, `.png`, `.jpeg`, `.webp`, and `.bmp` formats.

---

## 📊 Model Performance

Evaluated on **624 unseen test X-ray images**:

| Metric | Value | Details |
| :--- | :--- | :--- |
| **Test Accuracy** | **88.46%** | High overall diagnostic accuracy |
| **AUC-ROC Score** | **0.9468** | Strong class separability |
| **Training Set** | **5,856 images** | Kaggle Chest X-Ray Dataset (Paul Mooney) |
| **Input Tensor** | `224 × 224 × 3` | RGB preprocessed image tensor |

---

## 🧬 Model Architecture

The deep learning pipeline uses **VGG16** pre-trained on ImageNet as a feature extractor combined with custom classification layers:

```
[ Input: 224x224x3 Image ]
           │
           ▼
[ VGG16 Base Model (Frozen) ]  ───> 13 Conv Layers + 5 MaxPool Layers
           │
           ▼
[ GlobalAveragePooling2D ]    ───> Converts feature maps to 512-dim vector
           │
           ▼
[ Dense Layer (128 units) ]    ───> ReLU activation
           │
           ▼
[ Dropout (0.5) ]             ───> Prevents overfitting
           │
           ▼
[ Output Layer (1 unit) ]     ───> Sigmoid activation (Binary Probability)
```

---

## 📁 Project Structure

```
PneumoAI/
│
├── app.py                  # Flask backend server & routes
├── pneumonia_model.keras   # Pre-trained VGG16 model weights
├── requirements.txt        # Python package dependencies
├── README.md               # Documentation & setup guide
│
├── static/
│   ├── css/
│   │   └── app.css         # Modern design system & UI tokens
│   ├── js/                 # Client-side scripts
│   └── uploads/            # Upload workspace directory
│
└── templates/
    ├── base.html           # Master HTML layout & navigation
    ├── home.html           # Landing page with feature highlights
    ├── predict.html        # Interactive X-ray analyzer page
    └── about.html          # Technical & model card information
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/aakash2266/pneumonia-app.git
cd pneumonia-app
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Renders the home landing page |
| `/predict` | `GET` | Renders the interactive X-ray predictor UI |
| `/about` | `GET` | Renders model card and dataset specifications |
| `/analyze` | `POST` | Accepts image file payload and returns JSON prediction |

### Sample `/analyze` Response
```json
{
  "result": "Pneumonia Detected",
  "result_class": "danger",
  "confidence": "97.71%",
  "recommendation": "Please consult a healthcare professional immediately for proper clinical diagnosis and treatment guidance."
}
```

---

## ⚠️ Medical Disclaimer

> **IMPORTANT NOTICE**
> 
> **PneumoAI is built strictly for educational, research, and demonstration purposes.**
> - It is **NOT** a certified medical device and must **NOT** be used to make clinical decisions or replace professional medical advice.
> - Always consult qualified healthcare professionals for medical diagnosis and treatment.

---

## 👨‍💻 Developer & Acknowledgments

- **Developer**: [Aakash](https://github.com/aakash2266)
- **Dataset**: Kaggle Chest X-Ray Images (Pneumonia) by Paul Mooney
- **Model Architecture**: VGG16 by Visual Geometry Group (Oxford)