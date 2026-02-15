"""
import numpy as np
from urllib.request import urlretrieve
import matplotlib.pyplot as ml
import gzip
import os
import cv2 

PIXELS= 28
url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def download():
    os.makedirs("mnist", exist_ok=True)
    for f in files.values():
        path = os.path.join("mnist", f)
        if not os.path.exists(path):
            urlretrieve(url + f, path)

download()

def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, PIXELS * PIXELS)

def load_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

X_train = load_images("mnist/" + files["train_images"])
y_train = load_labels("mnist/" + files["train_labels"])
X_test = load_images("mnist/" + files["test_images"])
y_test = load_labels("mnist/" + files["test_labels"])

X_train = X_train / 255.0
X_test = X_test / 255.0

INPUT = PIXELS*PIXELS
HIDDEN1 = 256
HIDDEN2= 128
OUTPUT = 10

rng = np.random.default_rng()
print(f"RNG: {rng}")
W1 = rng.normal(0, np.sqrt(2/INPUT), (INPUT, HIDDEN1))
W2 = rng.normal(0, np.sqrt(2/HIDDEN1), (HIDDEN1, HIDDEN2))
W3 = rng.normal(0, np.sqrt(2/HIDDEN2), (HIDDEN2, OUTPUT))
b1 = np.zeros((1, HIDDEN1))
b2 = np.zeros((1, HIDDEN2))
b3 = np.zeros((1, OUTPUT))

# fuctions
def relu(x):
    return np.maximum(0, x)
def relu_grad(x):
    return (x > 0).astype(float)
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)
def cross_entropy(pred, y):
    m = y.shape[0]
    return -np.log(pred[np.arange(m), y] + 1e-9).mean()

#training

LR = 0.005
EPOCHS = 120
BATCH = 128

for epoch in range(EPOCHS):
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]
    total_loss = 0

    for i in range(0, len(X_train), BATCH):

        X = X_train[i:i+BATCH]
        y = y_train[i:i+BATCH]

        Z1 = X @ W1 + b1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + b2
        A2= relu(Z2)

        Z3 = A2 @ W3 + b3
        Y_hat = softmax(Z3)

        loss = cross_entropy(Y_hat, y)
        total_loss += loss

        m = X.shape[0]
        #bp
        dZ3 = Y_hat
        dZ3[np.arange(m), y] -= 1
        dZ3 /= m
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * relu_grad(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_grad(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W1-=LR * dW1
        b1-=LR * db1
        W2-=LR * dW2
        b2-=LR * db2
        W3-=LR * dW3
        b3-=LR * db3

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}")
np.savez("model.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

'''def accuracy(X, y):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    preds = np.argmax(softmax(Z3), axis=1)
    return np.mean(preds == y)

print("Train acc:", accuracy(X_train, y_train))
print("Test acc:", accuracy(X_test, y_test))"""