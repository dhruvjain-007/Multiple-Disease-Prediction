import os
import sys
from flask import Flask, request, jsonify

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'Frontend')
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array

app = Flask(__name__)

disease_model = None

def get_disease_model():
    global disease_model
    if disease_model is None:
        disease_model = DiseaseModel()
        disease_model.load_xgboost(os.path.join(FRONTEND_DIR, 'model', 'xgboost_model.json'))
    return disease_model

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    dm = get_disease_model()
    return jsonify(list(dm.all_symptoms))

@app.route('/api/predict/symptom', methods=['POST'])
@app.route('/api/symptom', methods=['POST'])
def predict_symptom():
    data = request.json or {}
    symptoms = data.get('symptoms', [])
    dm = get_disease_model()
    X = prepare_symptoms_array(symptoms)
    prediction, prob = dm.predict(X)
    description = dm.describe_predicted_disease()
    precautions = dm.predicted_disease_precautions()
    return jsonify({
        'prediction': prediction,
        'probability': prob,
        'description': description,
        'precautions': precautions
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
