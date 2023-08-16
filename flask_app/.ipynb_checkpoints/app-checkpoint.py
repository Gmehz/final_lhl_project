import numpy as np
import joblib
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model, the vectorizer, and the encoder
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptom_description = request.form['symptom_description']
        
        # Print the received symptom
        print("Received Symptom:", symptom_description)

        
        # Transform the symptom description
        input_data = vectorizer.transform([symptom_description])
        
        # Get the encoded prediction
        encoded_result = model.predict(input_data)
        
        # Decode the prediction
        decoded_prediction = encoder.inverse_transform(encoded_result)[0]

        # Debugging print statements
        print("Encoded Result:", encoded_result)
        print("Decoded Prediction:", decoded_prediction)

        return render_template('index.html', prediction=decoded_prediction)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555, debug=True)