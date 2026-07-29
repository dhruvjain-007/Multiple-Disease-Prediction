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
        model = joblib.load(os.path.join(FRONTEND_DIR, 'models', 'diabetes_model.sav'))
    return model

@app.route('/api/predict/diabetes', methods=['POST'])
@app.route('/api/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json or {}
    m = get_model()
    name = data.get('Name', 'Patient')
    features = [[
        data.get('Pregnancies', 0),
        data.get('Glucose', 0),
        data.get('BloodPressure', 0),
        data.get('SkinThickness', 0),
        data.get('Insulin', 0),
        data.get('BMI', 0),
        data.get('DiabetesPedigreeFunction', 0),
        data.get('Age', 0)
    ]]
    pred = int(m.predict(features)[0])
    result = "Assessment indicates potential metabolic disorder." if pred == 1 else "Assessment shows no metabolic disorder detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(port=5002, debug=True)
