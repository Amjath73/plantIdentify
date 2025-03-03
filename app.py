from __future__ import division, print_function
import sys
import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img, img_to_array
from flask_cors import CORS


# Define a flask app
app = Flask(__name__)
CORS(app)

# Model path
MODEL_PATH = r"D:\Downloaded\model\20240921-2014-full-image-set-mobilenetv2-Adam.h5"

# Load model with the custom KerasLayer
model = load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

# Model loaded message
print('Model loaded. Check http://127.0.0.1:5000/')

# Define your class labels
class_labels = ['African Violet (Saintpaulia ionantha)', 'Aloe Vera',
       'Anthurium (Anthurium andraeanum)',
       'Areca Palm (Dypsis lutescens)',
       'Asparagus Fern (Asparagus setaceus)', 'Begonia (Begonia spp.)',
       'Bird of Paradise (Strelitzia reginae)',
       'Birds Nest Fern (Asplenium nidus)',
       'Boston Fern (Nephrolepis exaltata)', 'Calathea',
       'Cast Iron Plant (Aspidistra elatior)',
       'Chinese Money Plant (Pilea peperomioides)',
       'Chinese evergreen (Aglaonema)',
       'Christmas Cactus (Schlumbergera bridgesii)', 'Chrysanthemum',
       'Ctenanthe', 'Daffodils (Narcissus spp.)', 'Dracaena',
       'Dumb Cane (Dieffenbachia spp.)', 'Elephant Ear (Alocasia spp.)',
       'English Ivy (Hedera helix)', 'Hyacinth (Hyacinthus orientalis)',
       'Iron Cross begonia (Begonia masoniana)',
       'Jade plant (Crassula ovata)', 'Kalanchoe',
       'Lilium (Hemerocallis)',
       'Lily of the valley (Convallaria majalis)',
       'Money Tree (Pachira aquatica)',
       'Monstera Deliciosa (Monstera deliciosa)', 'Orchid',
       'Parlor Palm (Chamaedorea elegans)', 'Peace lily',
       'Poinsettia (Euphorbia pulcherrima)',
       'Polka Dot Plant (Hypoestes phyllostachya)',
       'Ponytail Palm (Beaucarnea recurvata)', 'Pothos (Ivy arum)',
       'Prayer Plant (Maranta leuconeura)',
       'Rattlesnake Plant (Calathea lancifolia)',
       'Rubber Plant (Ficus elastica)', 'Sago Palm (Cycas revoluta)',
       'Schefflera', 'Snake plant (Sanseviera)', 'Tradescantia', 'Tulip',
       'Venus Flytrap', 'Yucca', 'ZZ Plant (Zamioculcas zamiifolia)']  # Replace with actual labels

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Assuming your model expects normalized input
    preds = model.predict(x)
    return preds

def custom_decode_predictions(preds, class_labels, top=1):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]  # Get indices of top predictions
        top_labels = [(class_labels[i], pred[i]) for i in top_indices]
        results.append(top_labels)
    return results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'house_plant_photo', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        pred_class = custom_decode_predictions(preds, class_labels, top=1)
        result = str(pred_class[0][0][0])
        return result

if __name__ == '__main__':
    app.run(debug=True)