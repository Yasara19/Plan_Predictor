import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("Plan_predictor.h5")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        hemoglobin = data['Hemoglobin_Level']
        glucose = data['Fasting_Glucose_Level']
        bmi = data['BMI_Value']

        # Make a prediction
        prediction = model.predict(np.array([[hemoglobin, glucose, bmi]]))

        # Convert the prediction to a list
        results = prediction.tolist()

        class_index = results[0].index(max(results[0]))

        return jsonify({'prediction': class_index})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
