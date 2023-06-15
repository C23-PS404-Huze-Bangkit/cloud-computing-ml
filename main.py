import os
import tensorflow as tf
import keras.utils as image
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

classes = (
    "American Shorthair",
    "Basset hound",
    "Beagle",
    "Bengal",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "English cocker spaniel",
    "Japanese chin",
    "Maine Coon",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Samoyed",
    "Scottish fold",
    "Siamese",
    "Sphynx"
)

model = tf.keras.models.load_model('Inceptionv3_datafix.h5')

@app.route('/', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    img = img.resize((299, 299))

    input_data = image.img_to_array(img)
    input_data = (input_data / 255.0)
    input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    result = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }

    return jsonify(result)

if __name__ == '__main__':
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
