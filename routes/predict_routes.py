import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from PIL import Image
import io
from ml_model import model, CLASS_NAMES
from dependencies import get_current_user
import models

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WEBP images are accepted")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return {
        "disease": predicted_class,
        "confidence": round(confidence * 100, 2),
    }
