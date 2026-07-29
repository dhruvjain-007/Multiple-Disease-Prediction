import os
import sys
import joblib
from flask import Flask, request, jsonify

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'Frontend')

app = Flask(__name__)

model = None

def get_model():
    global model
    if model is None:
        model = joblib.load(os.path.join(FRONTEND_DIR, 'models', 'heart_disease_model.sav'))
    return model

@app.route('/api/predict/heart', methods=['POST'])
@app.route('/api/heart', methods=['POST'])
def predict_heart():
    data = request.json or {}
    m = get_model()
    name = data.get('name', 'Patient')
    features = [[
        data.get('age', 0), data.get('sex', 0), data.get('cp', 0),
        data.get('trestbps', 0), data.get('chol', 0), data.get('fbs', 0),
        data.get('restecg', 0), data.get('thalach', 0), data.get('exang', 0),
        data.get('oldpeak', 0), data.get('slope', 0), data.get('ca', 0),
        data.get('thal', 0)
    ]]
    pred = int(m.predict(features)[0])
    result = "Evaluation suggests cardiovascular risk present." if pred == 1 else "Evaluation indicates no cardiovascular risk detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(port=5003, debug=True)
