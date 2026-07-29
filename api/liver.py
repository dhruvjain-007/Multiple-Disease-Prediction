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
        model = joblib.load(os.path.join(FRONTEND_DIR, 'models', 'liver_model.sav'))
    return model

@app.route('/api/predict/liver', methods=['POST'])
@app.route('/api/liver', methods=['POST'])
def predict_liver():
    data = request.json or {}
    m = get_model()
    name = data.get('name', 'Patient')
    features = [[
        data.get('sex', 0), data.get('age', 0), data.get('tb', 0),
        data.get('db', 0), data.get('alp', 0), data.get('alt', 0),
        data.get('ast', 0), data.get('tp', 0), data.get('alb', 0),
        data.get('agr', 0)
    ]]
    pred = int(m.predict(features)[0])
    result = "Analysis indicates potential liver dysfunction." if pred == 1 else "Analysis shows normal liver function."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(port=5005, debug=True)
