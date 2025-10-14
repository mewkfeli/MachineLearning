from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image, ImageOps
import io
import os
from tensorflow.keras.models import load_model

np.set_printoptions(suppress=True)

app = FastAPI(title="Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []


@app.on_event("startup")
async def load_model_and_labels():
    global model, class_names

    model_path = "keras_model.h5"
    labels_path = "labels.txt"

    model = load_model(model_path, compile=False)
    model.make_predict_function()

    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    if class_names and "Classes" in class_names[0]:
        class_names = class_names[1:]


def predict_image(image_data):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    class_name, confidence = predict_image(image_data)

    return {
        "class_name": class_name,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "status": "success"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": len(class_names) > 0,
        "classes": class_names if class_names else []
    }


@app.get("/")
async def root():
    return {"message": "Image Classification API is running!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)