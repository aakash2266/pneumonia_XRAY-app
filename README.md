# PneumoAI 🫁  
**Deep Learning for Pneumonia Detection from Chest X-Ray Images**

---

## 📌 Project Overview

**PneumoAI** is an end-to-end deep learning application that leverages **Convolutional Neural Networks (CNNs)** and **transfer learning** to detect **pneumonia** from chest X-ray images.  
The project demonstrates how artificial intelligence can assist medical professionals by providing **rapid, AI-powered preliminary screening** of radiographic images.

Pneumonia is a serious lung infection affecting millions worldwide. Early detection is critical, and this project explores how deep learning can support diagnostic workflows in healthcare.

---

## 🧠 Technical Architecture

- **Model**: VGG16 (Transfer Learning)
- **Input Size**: `224 × 224 × 3` RGB images
- **Framework**: TensorFlow / Keras
- **Backend**: Flask
- **Frontend**: HTML, CSS, Bootstrap 5, JavaScript
- **Deployment**: Flask Web Application

---

## 📊 Model Performance

Evaluated on **624 test images**:

| Metric | Value |
|------|------|
| **Test Accuracy** | **88.46%** |
| **AUC Score** | **0.9468** |
| **Training Images** | **5,856** |

---

## 🛠️ Technology Stack

### Deep Learning & Machine Learning
- Python 3.12  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Scikit-learn  

### Web Development
- Flask  
- Bootstrap 5  
- HTML5 / CSS3  
- JavaScript  

---

## 🧬 Model Architecture

### Transfer Learning with VGG16

The project uses **VGG16**, a deep convolutional neural network pre-trained on **ImageNet (1.4M images, 1000 classes)**.  
Transfer learning enables reuse of powerful feature extractors learned from large-scale datasets and fine-tuning them for pneumonia detection.

#### Network Structure
- VGG16 Base Model (Frozen weights)
  - 13 Convolutional layers  
  - 5 Max Pooling layers  
- Global Average Pooling  
- Dense Layer (128 units, ReLU)  
- Dropout (0.5)  
- Output Layer (1 unit, Sigmoid)

---

## 🗂️ Dataset Information

**Chest X-Ray Images (Pneumonia)**  
- **Source**: Kaggle – Paul Mooney  
- **Total Images**: 5,856  
- **Image Format**: JPEG  

### Class Distribution
- **NORMAL**: 1,583 images (27%)  
- **PNEUMONIA**: 4,273 images (73%)  

### Dataset Split
- **Training**: 5,216 images  
- **Validation**: 16 images  
- **Test**: 624 images  

---

## 🔄 Data Preprocessing

- Image resizing to `224 × 224`
- Pixel normalization to `[0, 1]`
- Data augmentation:
  - Rotation  
  - Width & height shifting  
  - Zoom  
  - Horizontal flip  
- Class weighting to handle imbalance  

---

## ⚙️ Training Configuration

- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Batch Size**: 32  
- **Epochs**: 10  
- **Early Stopping**:
  - Monitored on validation loss  
  - Patience = 5 epochs  

---

## 🔍 How It Works

### Prediction Pipeline

1. **Image Upload**  
   User uploads a chest X-ray via the web interface.

2. **Preprocessing**  
   Image is resized and normalized.

3. **Feature Extraction**  
   VGG16 extracts high-level spatial features.

4. **Classification**  
   Custom dense layers predict pneumonia probability.

5. **Result Display**  
   Prediction with confidence score is shown to the user.

---

## 📈 Understanding the Output

The model outputs a probability value between **0 and 1**:

- **Probability > 0.5** → **PNEUMONIA**
- **Probability ≤ 0.5** → **NORMAL**

The confidence score indicates how strongly the model supports its prediction.

---

## ⚠️ Important Disclaimer

> **This project is for educational and demonstration purposes only.**

- ❌ Not a substitute for professional medical diagnosis  
- ⚠️ May produce false positives or false negatives  
- 🩺 Always consult qualified healthcare professionals  
- 📸 Results depend on image quality  
- 🔒 No patient data is stored or transmitted  

If you have medical concerns, seek immediate professional medical attention.

---

## 🎯 Project Goals

- Demonstrate real-world application of deep learning in healthcare  
- Showcase CNNs and transfer learning techniques  
- Build a complete ML pipeline (data → model → deployment)  
- Create an accessible and user-friendly web interface  
- Explore AI-assisted medical image analysis  

---

## 🚀 Future Improvements

- Integrate **Grad-CAM** for model interpretability  
- Extend to **multi-class classification** (bacterial vs viral pneumonia)  
- Use **ensemble models** for improved robustness  
- Add **DICOM image support**  
- Develop a **mobile application**  

---

## 📁 Project Structure (High-Level)

PneumoAI/
│
├── app.py # Flask application
├── model/
│ └── pneumonia_model.keras
├── templates/
│ ├── home.html
│ ├── predict.html
│ └── about.html
├── static/
│ ├── css/
│ ├── js/
│ └── uploads/
├── notebooks/
│ └── model_training.ipynb
├── requirements.txt
└── README.md