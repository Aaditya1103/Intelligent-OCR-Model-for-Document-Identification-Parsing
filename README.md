# Intelligent-OCR-Model-for-Document-Identification-Parsing
---

## Table of Contents
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation and Setup](#installation-and-setup)
4. [Usage Instructions](#usage-instructions)
    - [Aadhaar OCR Tool](#aadhaar-ocr-tool)
    - [PAN Face Detector and OCR](#pan-face-detector-and-ocr)
5. [Example Outputs](#example-outputs)
6. [Folder Structure](#folder-structure)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)
10. [Author](#author)

---

## Overview
This project focuses on training a handwriting recognition model using Convolutional Neural Networks (CNNs) implemented with TensorFlow/Keras. The model is trained on labeled grayscale image data and predicts handwritten digits with high accuracy.

### Dataset Preparation
The dataset consists of grayscale images of handwritten digits, categorized into train, validation, and test splits. Each image is resized to 28x28 pixels and normalized to a [0,1] range. Labels are extracted from filenames and converted into a one-hot encoded format.

The dataset is stored in `.npz` format to optimize loading speed and memory usage.

### Model Architecture
The model follows a deep learning-based CNN approach with the following architecture:
- **Conv2D (32 filters, 3x3 kernel, ReLU activation, input: (28,28,1))**
- **MaxPooling2D (2x2 pooling window)**
- **Conv2D (64 filters, 3x3 kernel, ReLU activation)**
- **MaxPooling2D (2x2 pooling window)**
- **Flatten Layer**
- **Dense (128 neurons, ReLU activation)**
- **Dense (10 neurons, Softmax activation for classification)**
The model is compiled using Adam optimizer and categorical cross-entropy loss.
---

## Installation
To set up the environment and dependencies, follow these steps:
Ensure that TensorFlow and OpenCV are installed:
- **Clone Repository**:

   ```bash
  git clone https://github.com/your-repo/handwriting-recognition.git
  cd handwriting-recognition
   
- **Install Dependencies**:
  
   `pip install -r requirements.txt`
- **Ensure that TensorFlow and OpenCV are installed**:
  
   `pip install tensorflow opencv-python numpy scikit-learn`
---

## Installation and Setup

### Prerequisites
1. **Python 3.8+**: Install Python from [here](https://www.python.org/downloads/).
2. **Tesseract OCR**:
   - Install Tesseract OCR on your system:
     - **Windows**: [Download Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
     - **Linux**: Use `sudo apt install tesseract-ocr`.
     - **macOS**: Use Homebrew: `brew install tesseract`.
3. **Install Python Libraries**:
   - Install the required libraries using pip:
     ```bash
     pip install opencv-python-headless matplotlib pytesseract pillow easyocr
     ```
## Directory Structure
    handwriting-recognition/
    │── data/
    │   ├── train.npz
    │   ├── val.npz
    │   ├── test.npz
    │
    │── config/
    │   ├── rec_chinese_lite_train.yaml
    │
    │── src/
    │   ├── data_preprocessing.py
    │   ├── model.py
    │   ├── evaluation.py
    │
    │── namesplit.py
    │── ocr.py    
    │── name_identify.py
    │── name_list.xlx
    │── README.md
    │── requirements.txt
---
## Troubleshooting
**Issue**: Model training is slow or crashes due to memory issues.
- **Solution**: Reduce batch size or use GPU acceleration (TensorFlow with CUDA).

**Issue**: Test accuracy is lower than expected.
- **Solution**:Check if labels are correctly encoded, increase dataset size, or tune hyperparameters.

**Issue**: `ModuleNotFoundError` when running scripts.
- **Solution**: Run `pip install -r requirements.txt` to ensure all dependencies are installed.
---
     
## Future Improvements
1. **Data Augmentation**: Introduce transformations (rotation, scaling, noise) to enhance generalization.
2. **Hyperparameter Optimization**: Experiment with different architectures, activation functions, and optimizers.
3. **Deployment**: Convert the model to TensorFlow Lite for mobile or web applications.
4. **Additional Classes**: Extend the dataset beyond digits to include letters and symbols.
