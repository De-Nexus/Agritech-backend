import os
import gdown
import tensorflow as tf

MODEL_PATH = "agritech_model.keras"
GDRIVE_FILE_ID = "15KXfewEtyRMWt_mCV2-oDUB0fruMhKDZ"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Corn - Cercospora Leaf Spot / Gray Leaf Spot",
    "Corn - Common Rust",
    "Corn - Healthy",
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Target Spot",
    "Tomato - Healthy",
]
