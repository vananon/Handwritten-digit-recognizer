# Handwritten Digit Recognizer 

An interactive web application that allows users to draw a handwritten digit and receive real-time classification using a deep learning model implemented **from scratch with NumPy**.  

This project bridges **frontend engineering** (React + Figma design) with **neural network implementation**, showcasing both software craftsmanship and machine learning fundamentals.

---

## Demo
[Click here](PUT-YOUR-LINK-HERE)  

---

## Features
- Interactive canvas for digit input (React + HTML5 Canvas API)  
- Real-time probability distribution across digits (0–9)  
- Neural network implemented from scratch (NumPy)  
- Fast backend powered by FastAPI + OpenCV  
- Trained on MNIST dataset  

---

## Architecture Overview
**Frontend:**
- React  
- TypeScript  
- Vite  
- HTML5 Canvas API  
- CSS  

**Backend:**
- FastAPI  
- NumPy  
- OpenCV  
- Base64  
- CORS  

**Flow Diagram:**  
Canvas → Image Preprocessing → Backend → Neural Network → Prediction

---

## Machine Learning Model
**Dataset:** MNIST  

**Custom implementation of:**
- Forward propagation  
- Backpropagation  
- Gradient descent  
- ReLU and Softmax activation  
- Cross-entropy loss  

**Architecture:**  
- Input layer: 784 neurons (28×28 pixels)  
- Hidden layers: 256 → 128 neurons  
- Output layer: 10 neurons (digits 0–9)  

**Training Details:**  
- Learning rate: 0.005  
- Epochs: 120  
- Batch size: 128  
- Optimizer: Mini-batch Gradient Descent  
- Loss function: Cross-Entropy  
- Activations: ReLU (hidden), Softmax (output)  

---

## Results
- Accuracy: ~95% on test set (from MNIST's dataset)  

--- 


## Installation

Clone the repository:
```bash
git clone https://github.com/vananon/Handwritten-digit-recognizer
cd Handwritten-digit-recognizer
```
Run frontend 
```bash
npm run dev
```

Run backend
```bash 
cd backend
uvicorn main:app --reload 
```