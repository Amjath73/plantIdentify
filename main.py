import os
import gdown
import shutil
import numpy as np
import uvicorn
import tensorflow as tf
import tensorflow_hub as hub
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://plant-identify.onrender.com/"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive Model File ID (Extracted from Link)
MODEL_FILE_ID = "1uCrx2dzeaYxoqatYgfA4dB4WYR8QaUVA"  # Replace with actual ID
MODEL_PATH = "model.h5"

# Function to download model if not available
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Verify if model file exists and is valid
    if os.path.exists(MODEL_PATH):
        try:
            _ = load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
            print("✅ Model downloaded and verified successfully!")
        except Exception as e:
            print(f"❌ Model corrupted: {e}")
            os.remove(MODEL_PATH)  # Delete invalid file
            download_model()  # Retry download
    else:
        print("❌ Model download failed! Exiting...")
        exit(1)

# Ensure model is available before loading
download_model()

# Load the model
print("Loading model...")
model = load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
print("✅ Model loaded successfully!")

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
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize input
    preds = model.predict(x)
    return preds

# Decode predictions
def custom_decode_predictions(preds, class_labels, top=1):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        top_labels = [(class_labels[i], float(pred[i])) for i in top_indices]
        results.append(top_labels)
    return results

# API Endpoints
@app.get("/")
def home():
    return {"message": "FastAPI Plant Prediction API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        preds = model_predict(temp_file_path, model)
        pred_class = custom_decode_predictions(preds, class_labels, top=1)
        return JSONResponse({"prediction": pred_class[0][0][0], "confidence": pred_class[0][0][1]})
    finally:
        os.remove(temp_file_path)  # Clean up temp file

# Run FastAPI
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting FastAPI on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
