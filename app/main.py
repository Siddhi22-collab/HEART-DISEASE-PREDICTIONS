# flask_app/app.py
from flask import Flask, render_template, request
import pickle
import json
import pandas as pd

app = Flask(__name__)

# Load model
with open('../models/Heart_Disease.pkl', 'rb') as f:
    model = pickle.load(f)

# Load features
with open('../models/features.json', 'r') as f:
    features = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = {}
        for feature in features:
            user_input[feature] = request.form[feature]
        
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        prediction = "Presence of Heart Disease" if prediction == 1 else "No Heart Disease"
    
    return render_template('index.html', features=features, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
