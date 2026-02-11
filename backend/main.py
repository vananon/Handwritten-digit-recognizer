from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import base64
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

data = np.load("model.npz")

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]
W3 = data["W3"]
b3 = data["b3"]

#activation fuctions

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

#endpoint

@app.post("/predict")
async def predict(payload: dict):
    img_base64 = payload["image"].split(",")[1]
    img_bytes = base64.b64decode(img_base64)

    img = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    # Preprocess
    img = cv2.GaussianBlur(img, (5, 5), 0)

    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(img)

    if coords is None:
        return {
            "prediction": -1,
            "probabilities": []
        }

    bx, by, bw, bh = cv2.boundingRect(coords)
    img = img[by:by+bh, bx:bx+bw]

    img = cv2.resize(img, (20, 20))
    pad = 4
    img = cv2.copyMakeBorder(
        img,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=0
    )


    #normalize
    img = img / 255.0
    img = img.reshape(1, 784)

    Z1 = img @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3

    probs = softmax(Z3)

    return {
        "prediction": int(np.argmax(probs)),
        "probabilities": probs[0].tolist()
    }
