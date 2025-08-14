import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# -------------------
# Flask App Setup
# -------------------
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------
# Load the trained model
# -------------------
MODEL_PATH = 'model/vegetable_classifier.h5'  # Change if your model filename is different
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------
# Class Labels Mapping
# -------------------
class_map = {
    0: 'Bean', 
    1: 'Bitter Gourd', 
    2: 'Bottle Gourd', 
    3: 'Brinjal',
    4: 'Broccoli', 
    5: 'Cabbage', 
    6: 'Capsicum', 
    7: 'Carrot',
    8: 'Cauliflower', 
    9: 'Cucumber', 
    10: 'Papaya', 
    11: 'Potato'
}

# -------------------
# Routes
# -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files.get('image')
        if not file:
            return render_template('index.html', label="No file selected", image_url=None)

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        label = predict_image(filepath)

        return render_template('index.html', label=label, image_url=filepath)

    return render_template('index.html', label=None, image_url=None)

# -------------------
# Prediction Function
# -------------------
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds) * 100

        return f"{class_map.get(predicted_class, 'Unknown')} ({confidence:.2f}%)"
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------
# Main Entry Point
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
