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
        model = joblib.load(os.path.join(FRONTEND_DIR, 'models', 'hepititisc_model.sav'))
    return model

@app.route('/api/predict/hepatitis', methods=['POST'])
@app.route('/api/hepatitis', methods=['POST'])
def predict_hepatitis():
    data = request.json or {}
    m = get_model()
    name = data.get('name', 'Patient')
    df = pd.DataFrame({
        'Age': [data.get('age', 0)],
        'Sex': [data.get('sex', 1)],
        'ALB': [data.get('alb', 0)],
        'ALP': [data.get('alp', 0)],
        'ALT': [data.get('alt', 0)],
        'AST': [data.get('ast', 0)],
        'BIL': [data.get('bil', 0)],
        'CHE': [data.get('che', 0)],
        'CHOL': [data.get('chol', 0)],
        'CREA': [data.get('crea', 0)],
        'GGT': [data.get('ggt', 0)],
        'PROT': [data.get('prot', 0)]
    })
    pred = int(m.predict(df)[0])
    result = "Screening indicates potential hepatitis risk." if pred == 1 else "Screening shows no hepatitis detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(port=5006, debug=True)
