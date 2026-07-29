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
        model = joblib.load(os.path.join(FRONTEND_DIR, 'models', 'chronic_model.sav'))
    return model

@app.route('/api/predict/kidney', methods=['POST'])
@app.route('/api/kidney', methods=['POST'])
def predict_kidney():
    data = request.json or {}
    m = get_model()
    name = data.get('name', 'Patient')
    df = pd.DataFrame({
        'age': [data.get('age', 0)], 'bp': [data.get('bp', 0)], 'sg': [data.get('sg', 1.02)],
        'al': [data.get('al', 0)], 'su': [data.get('su', 0)], 'rbc': [data.get('rbc', 1)],
        'pc': [data.get('pc', 1)], 'pcc': [data.get('pcc', 0)], 'ba': [data.get('ba', 0)],
        'bgr': [data.get('bgr', 0)], 'bu': [data.get('bu', 0)], 'sc': [data.get('sc', 0)],
        'sod': [data.get('sod', 0)], 'pot': [data.get('pot', 0)], 'hemo': [data.get('hemo', 0)],
        'pcv': [data.get('pcv', 0)], 'wc': [data.get('wc', 0)], 'rc': [data.get('rc', 0)],
        'htn': [data.get('htn', 0)], 'dm': [data.get('dm', 0)], 'cad': [data.get('cad', 0)],
        'appet': [data.get('appet', 1)], 'pe': [data.get('pe', 0)], 'ane': [data.get('ane', 0)]
    })
    pred = int(m.predict(df)[0])
    result = "Monitoring indicates potential kidney dysfunction." if pred == 1 else "Monitoring shows normal kidney function."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(port=5008, debug=True)
