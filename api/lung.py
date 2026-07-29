import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'Frontend')

app = Flask(__name__)

model = None

def get_model():
    global model
    if model is None:
        model = joblib.load(os.path.join(FRONTEND_DIR, 'models', 'lung_cancer_model.sav'))
    return model

@app.route('/api/predict/lung', methods=['POST'])
@app.route('/api/lung', methods=['POST'])
def predict_lung():
    data = request.json or {}
    m = get_model()
    name = data.get('name', 'Patient')
    df = pd.DataFrame({
        'GENDER': [data.get('gender', 'Male')],
        'AGE': [data.get('age', 0)],
        'SMOKING': [data.get('smoking', 'NO')],
        'YELLOW_FINGERS': [data.get('yellow_fingers', 'NO')],
        'ANXIETY': [data.get('anxiety', 'NO')],
        'PEER_PRESSURE': [data.get('peer_pressure', 'NO')],
        'CHRONICDISEASE': [data.get('chronic_disease', 'NO')],
        'FATIGUE': [data.get('fatigue', 'NO')],
        'ALLERGY': [data.get('allergy', 'NO')],
        'WHEEZING': [data.get('wheezing', 'NO')],
        'ALCOHOLCONSUMING': [data.get('alcohol_consuming', 'NO')],
        'COUGHING': [data.get('coughing', 'NO')],
        'SHORTNESSOFBREATH': [data.get('shortness_of_breath', 'NO')],
        'SWALLOWINGDIFFICULTY': [data.get('swallowing_difficulty', 'NO')],
        'CHESTPAIN': [data.get('chest_pain', 'NO')]
    })
    df.replace({'NO': 1, 'YES': 2}, inplace=True)
    df.columns = df.columns.str.strip()
    pred = str(m.predict(df)[0])
    result = "Screening indicates potential pulmonary risk." if pred == 'YES' else "Screening shows no pulmonary risk detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(port=5007, debug=True)
