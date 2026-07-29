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
        model = joblib.load(os.path.join(FRONTEND_DIR, 'models', 'parkinsons_model.sav'))
    return model

@app.route('/api/predict/parkinson', methods=['POST'])
@app.route('/api/parkinson', methods=['POST'])
def predict_parkinson():
    data = request.json or {}
    m = get_model()
    name = data.get('name', 'Patient')
    features = [[
        data.get('fo',0), data.get('fhi',0), data.get('flo',0), data.get('jit',0),
        data.get('jitabs',0), data.get('rap',0), data.get('ppq',0), data.get('ddp',0),
        data.get('shim',0), data.get('shimdb',0), data.get('apq3',0), data.get('apq5',0),
        data.get('apq',0), data.get('dda',0), data.get('nhr',0), data.get('hnr',0),
        data.get('rpde',0), data.get('dfa',0), data.get('spr1',0), data.get('spr2',0),
        data.get('d2',0), data.get('ppe',0)
    ]]
    pred = int(m.predict(features)[0])
    result = "Screening indicates potential movement disorder." if pred == 1 else "Screening shows no movement disorder detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(port=5004, debug=True)
