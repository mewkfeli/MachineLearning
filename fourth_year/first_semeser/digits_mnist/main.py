from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import uvicorn

app = FastAPI(title="Определение цифры")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    try:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.load_weights('model11.weights.h5')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()
class_names = ['Ноль', "Единица", "Двойка", "Тройка", "Четвёрка",
               "Пятёрка", "Шестёрка", "Семёрка", "Восьмёрка", "Девятка"]

def preprocess_image(image_data):
    if image_data.mode != 'L':
        image_data = image_data.convert('L')

    image_data = image_data.resize((28, 28))
    img_array = np.array(image_data)

    img_array = 255 - img_array

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 28, 28)

    return img_array

@app.get("/")
async def root():
    return {"message": "MNIST Digit Recognition API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        all_probabilities = {
            class_names[i]: float(predictions[0][i]) for i in range(10)
        }

        return {
            "predicted_digit": int(predicted_class),
            "predicted_class": class_names[predicted_class],
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/predict_base64")
async def predict_base64(image_data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        base64_string = image_data.get("image")
        if not base64_string:
            raise HTTPException(status_code=400, detail="Нет изображения")

        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        all_probabilities = {
            class_names[i]: float(predictions[0][i]) for i in range(10)
        }

        return {
            "predicted_digit": int(predicted_class),
            "predicted_class": class_names[predicted_class],
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)