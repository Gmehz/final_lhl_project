import numpy as np
import joblib
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model and the vectorizer
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptom_description = request.form['symptom_description']
        
        # Transform the symptom description to be fed into the model
        input_data = vectorizer.transform([symptom_description])
        
        result = model.predict(input_data)

        # Assuming your model predicts the disease name directly
        prediction = result[0]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555, debug=True)
