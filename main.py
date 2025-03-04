import os
import uvicorn
import numpy as np
import shutil
import requests
import tensorflow_hub as hub
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Set model path and download if necessary
MODEL_URL = "https://drive.google.com/uc?export=download&id=1uCrx2dzeaYxoqatYgfA4dB4WYR8QaUVA"
MODEL_PATH = "/app/model/mobilenetv2.h5"

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"📥 Model not found locally. Downloading from {MODEL_URL} ...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print("✅ Model downloaded successfully!")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Railway!"}

# Load model
print(f"🔍 Model path: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevent app from crashing

# Class labels
class_labels = [
    'African Violet (Saintpaulia ionantha)', 'Aloe Vera', 'Anthurium (Anthurium andraeanum)', 
    'Areca Palm (Dypsis lutescens)', 'Asparagus Fern (Asparagus setaceus)', 
    'Begonia (Begonia spp.)', 'Bird of Paradise (Strelitzia reginae)', 
    'Birds Nest Fern (Asplenium nidus)', 'Boston Fern (Nephrolepis exaltata)', 
    'Calathea', 'Cast Iron Plant (Aspidistra elatior)', 'Chinese Money Plant (Pilea peperomioides)', 
    'Chinese evergreen (Aglaonema)', 'Christmas Cactus (Schlumbergera bridgesii)', 
    'Chrysanthemum', 'Ctenanthe', 'Daffodils (Narcissus spp.)', 'Dracaena', 
    'Dumb Cane (Dieffenbachia spp.)', 'Elephant Ear (Alocasia spp.)', 
    'English Ivy (Hedera helix)', 'Hyacinth (Hyacinthus orientalis)', 
    'Iron Cross begonia (Begonia masoniana)', 'Jade plant (Crassula ovata)', 
    'Kalanchoe', 'Lilium (Hemerocallis)', 'Lily of the valley (Convallaria majalis)', 
    'Money Tree (Pachira aquatica)', 'Monstera Deliciosa (Monstera deliciosa)', 
    'Orchid', 'Parlor Palm (Chamaedorea elegans)', 'Peace lily', 
    'Poinsettia (Euphorbia pulcherrima)', 'Polka Dot Plant (Hypoestes phyllostachya)', 
    'Ponytail Palm (Beaucarnea recurvata)', 'Pothos (Ivy arum)', 
    'Prayer Plant (Maranta leuconeura)', 'Rattlesnake Plant (Calathea lancifolia)', 
    'Rubber Plant (Ficus elastica)', 'Sago Palm (Cycas revoluta)', 'Schefflera', 
    'Snake plant (Sanseviera)', 'Tradescantia', 'Tulip', 'Venus Flytrap', 
    'Yucca', 'ZZ Plant (Zamioculcas zamiifolia)'
]

# Function to preprocess and predict
def model_predict(img_path, model):
    try:
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize input
        preds = model.predict(x)
        return preds
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

# Decode predictions
def custom_decode_predictions(preds, class_labels, top=1):
    if preds is None:
        return [["Error", 0.0]]
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        top_labels = [(class_labels[i], float(pred[i])) for i in top_indices]
        results.append(top_labels)
    return results

# API Endpoints
@app.get("/health")
def health_check():
    return {"status": "running", "message": "FastAPI Plant Prediction API is healthy!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded. Check your Railway setup."}, status_code=500)

    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        preds = model_predict(file_path, model)
        pred_class = custom_decode_predictions(preds, class_labels, top=1)
    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {e}"}, status_code=500)
    finally:
        os.remove(file_path)  # Clean up file after prediction

    return JSONResponse({"prediction": pred_class[0][0][0], "confidence": pred_class[0][0][1]})

# Run FastAPI
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Railway provides PORT dynamically
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
