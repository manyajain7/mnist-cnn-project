# MNIST CNN Project

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify handwritten digits from the **MNIST dataset**.

---

## Dataset
- **MNIST**: 28×28 grayscale images of handwritten digits (0–9)  
- **Training set**: 60,000 images  
- **Test set**: 10,000 images  

---

## Preprocessing
- Images are converted to **tensors** using `transforms.ToTensor()`  
- Normalized to mean=0.5, std=0.5  
- Data loaded in **batches of 64** for training/testing  

---

## Model Architecture
- **Convolutional Layers**:
  1. `Conv2d(1 → 32)` + ReLU
  2. `Conv2d(32 → 64)` + ReLU
- **Fully Connected Layers**:
  1. Flatten → 128 neurons + ReLU
  2. Output layer → 10 neurons (digits 0–9)  
- **Loss Function**: CrossEntropyLoss  
- **Optimizer**: Adam (learning rate = 0.001)  

---

## Training
- Number of epochs: 3  
- Batch size: 64  
- Training Loss decreased from ~0.14 → ~0.02  

---

## Results
- **Test Accuracy**: ~98.7%  
- **Sample Predictions**:

![Sample Predictions](predictions.png)

The model successfully predicts handwritten digits with high accuracy.

---

## How to Run
1. Install required packages:
```bash
pip install torch torchvision matplotlib
