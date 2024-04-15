# Updated Flask application code with fixes

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the trained TensorFlow model
model_path = r'C:\Users\abhis\AI_Photo_Tagger_Projec\model'
model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tag', methods=['POST'])
def tag_photo():
    try:
        # Get photo file from request
        photo = request.files['photo']

        # Save photo temporarily
        photo_path = 'temp_photo.jpg'
        photo.save(photo_path)

        # Preprocess photo
        photo_data = preprocess_photo(photo_path)

        # Perform inference with the model
        predictions = model.predict(np.expand_dims(photo_data, axis=0))

        # Decode predictions
        predicted_tags = decode_predictions(predictions)

        # Remove temporary photo file
        os.remove(photo_path)

        return jsonify({'tags': predicted_tags})
    except Exception as e:
        return jsonify({'error': str(e)})

def preprocess_photo(photo_path):
    # Implement photo preprocessing (e.g., resize, normalization)
    # Here's a simple placeholder implementation for demonstration
    return np.zeros((224, 224, 3))  # Placeholder numpy array

def decode_predictions(predictions):
    # Placeholder decoding logic
    # You should implement actual logic based on your model output
    return ['tag1', 'tag2', 'tag3']

if __name__ == '__main__':
    app.run(debug=True)
