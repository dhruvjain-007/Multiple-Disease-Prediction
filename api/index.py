import os
import sys
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'Frontend')
CODE_DIR = os.path.join(FRONTEND_DIR, 'code')

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from DiseaseModel import DiseaseModel
from helper import prepare_symptoms_array


app = Flask(__name__)

models = {}

def get_models():
    if not models:
        models_dir = os.path.join(FRONTEND_DIR, 'models')
        models['diabetes'] = joblib.load(os.path.join(models_dir, 'diabetes_model.sav'))
        models['heart'] = joblib.load(os.path.join(models_dir, 'heart_disease_model.sav'))
        models['parkinson'] = joblib.load(os.path.join(models_dir, 'parkinsons_model.sav'))
        models['lung_cancer'] = joblib.load(os.path.join(models_dir, 'lung_cancer_model.sav'))
        models['breast_cancer'] = joblib.load(os.path.join(models_dir, 'breast_cancer.sav'))
        models['chronic'] = joblib.load(os.path.join(models_dir, 'chronic_model.sav'))
        models['hepatitis'] = joblib.load(os.path.join(models_dir, 'hepititisc_model.sav'))
        models['liver'] = joblib.load(os.path.join(models_dir, 'liver_model.sav'))
        
        disease_model = DiseaseModel()
        disease_model.load_model()
        models['disease_model'] = disease_model
    return models

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diagno - AI Diagnostic System</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --primary: #38bdf8;
            --primary-hover: #0284c7;
            --accent: #818cf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #4ade80;
            --warning: #facc15;
            --danger: #f87171;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: radial-gradient(circle at top left, #1e1b4b, #0f172a, #090d16); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }
        header { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border-color); padding: 1.25rem 2rem; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
        header h1 { font-size: 1.6rem; font-weight: 700; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: flex; align-items: center; gap: 0.75rem; }
        .badge { background: rgba(56, 189, 248, 0.15); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.3); font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; font-weight: 600; }
        .layout { display: flex; flex: 1; }
        aside { width: 280px; background: rgba(15, 23, 42, 0.6); border-right: 1px solid var(--border-color); padding: 1.5rem 1rem; display: flex; flex-direction: column; gap: 0.5rem; }
        .nav-btn { background: transparent; border: 1px solid transparent; color: var(--text-muted); padding: 0.85rem 1rem; border-radius: 12px; text-align: left; font-size: 0.95rem; font-weight: 500; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; gap: 0.75rem; }
        .nav-btn:hover { background: rgba(255, 255, 255, 0.05); color: var(--text-main); }
        .nav-btn.active { background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)); border-color: rgba(56, 189, 248, 0.4); color: var(--primary); font-weight: 600; }
        main { flex: 1; padding: 2.5rem; max-width: 1100px; }
        .panel { display: none; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 20px; padding: 2rem; backdrop-filter: blur(16px); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3); animation: fadeIn 0.3s ease; }
        .panel.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .panel-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-main); }
        .panel-desc { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 2rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem; margin-bottom: 2rem; }
        .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
        .form-group label { font-size: 0.85rem; font-weight: 500; color: var(--text-muted); }
        .form-group input, .form-group select { background: rgba(15, 23, 42, 0.8); border: 1px solid var(--border-color); border-radius: 10px; padding: 0.75rem 1rem; color: var(--text-main); font-size: 0.95rem; outline: none; transition: border-color 0.2s ease; }
        .form-group input:focus, .form-group select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15); }
        .symptoms-selector { max-height: 280px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; background: rgba(15, 23, 42, 0.6); display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.6rem; margin-bottom: 1.5rem; }
        .symptom-item { display: flex; align-items: center; gap: 0.5rem; font-size: 0.88rem; color: var(--text-muted); cursor: pointer; padding: 0.3rem 0.5rem; border-radius: 6px; transition: background 0.15s ease; }
        .symptom-item:hover { background: rgba(255, 255, 255, 0.05); color: var(--text-main); }
        .symptom-item input[type="checkbox"] { accent-color: var(--primary); width: 16px; height: 16px; }
        .btn-submit { background: linear-gradient(135deg, var(--primary), var(--accent)); color: #0f172a; font-weight: 700; border: none; padding: 0.9rem 2rem; border-radius: 12px; font-size: 1rem; cursor: pointer; transition: transform 0.15s ease, box-shadow 0.15s ease; box-shadow: 0 10px 25px rgba(56, 189, 248, 0.25); }
        .btn-submit:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(56, 189, 248, 0.35); }
        .result-box { margin-top: 2rem; padding: 1.5rem; border-radius: 14px; background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border-color); display: none; }
        .result-box.active { display: block; }
        .result-header { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; }
        .result-desc { color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem; }
        .precautions-list { margin-top: 1rem; padding-left: 1.25rem; color: var(--text-main); }
        .precautions-list li { margin-bottom: 0.4rem; font-size: 0.9rem; }
        .search-wrapper { position: relative; margin-bottom: 1rem; }
        .autocomplete-dropdown { position: absolute; top: 100%; left: 0; right: 0; background: #1e293b; border: 1px solid var(--primary); border-radius: 12px; max-height: 240px; overflow-y: auto; z-index: 1000; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6); display: none; margin-top: 4px; }
        .autocomplete-dropdown.active { display: block; }
        .dropdown-item { padding: 0.75rem 1rem; color: var(--text-main); cursor: pointer; font-size: 0.9rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); transition: background 0.15s ease; display: flex; justify-content: space-between; align-items: center; }
        .dropdown-item:hover, .dropdown-item.active { background: rgba(56, 189, 248, 0.15); color: var(--primary); }
        .dropdown-item .add-tag { font-size: 0.75rem; background: rgba(56, 189, 248, 0.2); padding: 0.2rem 0.5rem; border-radius: 6px; color: var(--primary); font-weight: 600; }
        .selected-chips-box { margin-bottom: 1.25rem; }
        .selected-chips-header { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; }
        .selected-chips-container { display: flex; flex-wrap: wrap; gap: 0.5rem; padding: 0.75rem; background: rgba(15, 23, 42, 0.6); border: 1px solid var(--border-color); border-radius: 12px; min-height: 52px; align-items: center; }
        .chip { background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)); border: 1px solid rgba(56, 189, 248, 0.4); color: var(--text-main); padding: 0.35rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; display: flex; align-items: center; gap: 0.5rem; animation: fadeIn 0.15s ease; }
        .chip-remove { background: rgba(248, 113, 113, 0.2); color: var(--danger); border-radius: 50%; width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center; cursor: pointer; font-size: 0.75rem; line-height: 1; transition: background 0.15s ease; }
        .chip-remove:hover { background: var(--danger); color: #fff; }
        .empty-chips-msg { color: var(--text-muted); font-size: 0.85rem; font-style: italic; }
        .search-input { width: 100%; padding: 0.75rem 1rem; border-radius: 10px; background: rgba(15, 23, 42, 0.8); border: 1px solid var(--border-color); color: var(--text-main); outline: none; }
        @media (max-width: 768px) { .layout { flex-direction: column; } aside { width: 100%; border-right: none; border-bottom: 1px solid var(--border-color); } main { padding: 1.25rem; } }
    </style>
</head>
<body>
    <header>
        <h1><span>🩺</span> Diagno</h1>
        <span class="badge">AI Health Diagnostic System</span>
    </header>



    <div class="layout">
        <aside>
            <button class="nav-btn active" onclick="showTab('symptom')">🔍 Symptom Analysis</button>
            <button class="nav-btn" onclick="showTab('diabetes')">🩸 Metabolic Check</button>
            <button class="nav-btn" onclick="showTab('heart')">❤️ Cardiovascular Risk</button>
            <button class="nav-btn" onclick="showTab('parkinson')">🧠 Movement Disorder</button>
            <button class="nav-btn" onclick="showTab('liver')">🧪 Liver Function</button>
            <button class="nav-btn" onclick="showTab('hepatitis')">🔬 Hepatitis Screening</button>
            <button class="nav-btn" onclick="showTab('lung')">🫁 Pulmonary Check</button>
            <button class="nav-btn" onclick="showTab('kidney')">💧 Kidney Monitor</button>
        </aside>

        <main>
            <!-- 1. Symptom Analysis -->
            <div id="tab-symptom" class="panel active">
                <div class="panel-title">AI Symptom Analysis</div>
                <div class="panel-desc">Search and select observed symptoms to evaluate potential health conditions.</div>
                
                <div class="search-wrapper">
                    <input type="text" id="symptom-search" class="search-input" placeholder="🔍 Type a symptom (e.g. headache, fever, itching)..." autocomplete="off" oninput="onSearchInput(this.value)" onfocus="onSearchInput(this.value)">
                    <div id="autocomplete-dropdown" class="autocomplete-dropdown"></div>
                </div>

                <div class="selected-chips-box">
                    <div class="selected-chips-header">
                        <span>Selected Symptoms (<span id="selected-count">0</span>)</span>
                        <button type="button" onclick="clearAllSymptoms()" style="background: none; border: none; color: var(--danger); font-size: 0.8rem; cursor: pointer;">Clear All</button>
                    </div>
                    <div id="selected-chips" class="selected-chips-container">
                        <span class="empty-chips-msg">No symptoms selected yet. Type above or select from quick list below.</span>
                    </div>
                </div>

                <details style="margin-bottom: 1.5rem;" open>
                    <summary style="cursor: pointer; color: var(--primary); font-weight: 500; font-size: 0.9rem; margin-bottom: 0.75rem;">Browse All Symptoms Grid (133)</summary>
                    <div class="symptoms-selector" id="symptoms-list"></div>
                </details>

                <button class="btn-submit" onclick="submitSymptomAnalysis()">Analyze Symptoms</button>

                <div id="res-symptom" class="result-box">
                    <div id="res-symptom-header" class="result-header"></div>
                    <div id="res-symptom-desc" class="result-desc"></div>
                    <ul id="res-symptom-precautions" class="precautions-list"></ul>
                </div>
            </div>


            <!-- 2. Metabolic / Diabetes -->
            <div id="tab-diabetes" class="panel">
                <div class="panel-title">Metabolic Disorder Check (Diabetes)</div>
                <div class="panel-desc">Enter physiological parameters to assess diabetes risk.</div>
                <div class="form-grid">
                    <div class="form-group"><label>Patient Name</label><input type="text" id="db-name" placeholder="John Doe"></div>
                    <div class="form-group"><label>Pregnancies</label><input type="number" id="db-preg" value="0"></div>
                    <div class="form-group"><label>Glucose Level</label><input type="number" id="db-glu" value="120"></div>
                    <div class="form-group"><label>Blood Pressure</label><input type="number" id="db-bp" value="70"></div>
                    <div class="form-group"><label>Skin Thickness</label><input type="number" id="db-skin" value="20"></div>
                    <div class="form-group"><label>Insulin</label><input type="number" id="db-ins" value="80"></div>
                    <div class="form-group"><label>BMI</label><input type="number" step="0.1" id="db-bmi" value="25.0"></div>
                    <div class="form-group"><label>Diabetes Pedigree Function</label><input type="number" step="0.01" id="db-ped" value="0.5"></div>
                    <div class="form-group"><label>Age</label><input type="number" id="db-age" value="30"></div>
                </div>
                <button class="btn-submit" onclick="submitDiabetes()">Run Metabolic Assessment</button>
                <div id="res-diabetes" class="result-box"></div>
            </div>

            <!-- 3. Cardiovascular -->
            <div id="tab-heart" class="panel">
                <div class="panel-title">Cardiovascular Risk Assessment</div>
                <div class="panel-desc">Evaluate cardiac health markers for risk prediction.</div>
                <div class="form-grid">
                    <div class="form-group"><label>Patient Name</label><input type="text" id="ht-name" placeholder="Jane Doe"></div>
                    <div class="form-group"><label>Age</label><input type="number" id="ht-age" value="45"></div>
                    <div class="form-group"><label>Sex</label><select id="ht-sex"><option value="1">Male</option><option value="0">Female</option></select></div>
                    <div class="form-group"><label>Chest Pain Type</label><select id="ht-cp"><option value="0">Typical Angina</option><option value="1">Atypical Angina</option><option value="2">Non-Anginal Pain</option><option value="3">Asymptomatic</option></select></div>
                    <div class="form-group"><label>Resting BP (mm Hg)</label><input type="number" id="ht-trestbps" value="120"></div>
                    <div class="form-group"><label>Serum Cholesterol (mg/dl)</label><input type="number" id="ht-chol" value="200"></div>
                    <div class="form-group"><label>Fasting Blood Sugar > 120 mg/dl</label><select id="ht-fbs"><option value="0">No</option><option value="1">Yes</option></select></div>
                    <div class="form-group"><label>Resting ECG</label><select id="ht-restecg"><option value="0">Normal</option><option value="1">ST-T Abnormality</option><option value="2">LV Hypertrophy</option></select></div>
                    <div class="form-group"><label>Max Heart Rate</label><input type="number" id="ht-thalach" value="150"></div>
                    <div class="form-group"><label>Exercise Induced Angina</label><select id="ht-exang"><option value="0">No</option><option value="1">Yes</option></select></div>
                    <div class="form-group"><label>ST Depression (Oldpeak)</label><input type="number" step="0.1" id="ht-oldpeak" value="1.0"></div>
                    <div class="form-group"><label>Slope of Peak Exercise ST</label><select id="ht-slope"><option value="0">Upsloping</option><option value="1">Flat</option><option value="2">Downsloping</option></select></div>
                    <div class="form-group"><label>Major Vessels (0-3)</label><input type="number" id="ht-ca" value="0"></div>
                    <div class="form-group"><label>Thalassemia</label><select id="ht-thal"><option value="0">Normal</option><option value="1">Fixed Defect</option><option value="2">Reversible Defect</option></select></div>
                </div>
                <button class="btn-submit" onclick="submitHeart()">Evaluate Heart Health</button>
                <div id="res-heart" class="result-box"></div>
            </div>

            <!-- 4. Movement Disorder -->
            <div id="tab-parkinson" class="panel">
                <div class="panel-title">Movement Disorder Screening</div>
                <div class="panel-desc">Assess vocal and motor frequency features for Parkinson's disease detection.</div>
                <div class="form-grid">
                    <div class="form-group"><label>Patient Name</label><input type="text" id="pk-name" placeholder="Patient Name"></div>
                    <div class="form-group"><label>MDVP:Fo (Hz)</label><input type="number" step="0.01" id="pk-fo" value="119.99"></div>
                    <div class="form-group"><label>MDVP:Fhi (Hz)</label><input type="number" step="0.01" id="pk-fhi" value="157.30"></div>
                    <div class="form-group"><label>MDVP:Flo (Hz)</label><input type="number" step="0.01" id="pk-flo" value="74.99"></div>
                    <div class="form-group"><label>MDVP:Jitter (%)</label><input type="number" step="0.0001" id="pk-jit" value="0.0078"></div>
                    <div class="form-group"><label>MDVP:Jitter (Abs)</label><input type="number" step="0.00001" id="pk-jitabs" value="0.00007"></div>
                    <div class="form-group"><label>MDVP:RAP</label><input type="number" step="0.0001" id="pk-rap" value="0.0037"></div>
                    <div class="form-group"><label>MDVP:PPQ</label><input type="number" step="0.0001" id="pk-ppq" value="0.0055"></div>
                    <div class="form-group"><label>Jitter:DDP</label><input type="number" step="0.0001" id="pk-ddp" value="0.0110"></div>
                    <div class="form-group"><label>MDVP:Shimmer</label><input type="number" step="0.001" id="pk-shim" value="0.0437"></div>
                    <div class="form-group"><label>MDVP:Shimmer (dB)</label><input type="number" step="0.01" id="pk-shimdb" value="0.4260"></div>
                    <div class="form-group"><label>Shimmer:APQ3</label><input type="number" step="0.001" id="pk-apq3" value="0.0218"></div>
                    <div class="form-group"><label>Shimmer:APQ5</label><input type="number" step="0.001" id="pk-apq5" value="0.0313"></div>
                    <div class="form-group"><label>MDVP:APQ</label><input type="number" step="0.001" id="pk-apq" value="0.0297"></div>
                    <div class="form-group"><label>Shimmer:DDA</label><input type="number" step="0.001" id="pk-dda" value="0.0654"></div>
                    <div class="form-group"><label>NHR</label><input type="number" step="0.001" id="pk-nhr" value="0.0221"></div>
                    <div class="form-group"><label>HNR</label><input type="number" step="0.1" id="pk-hnr" value="21.03"></div>
                    <div class="form-group"><label>RPDE</label><input type="number" step="0.01" id="pk-rpde" value="0.4147"></div>
                    <div class="form-group"><label>DFA</label><input type="number" step="0.01" id="pk-dfa" value="0.8152"></div>
                    <div class="form-group"><label>spread1</label><input type="number" step="0.01" id="pk-spr1" value="-4.8100"></div>
                    <div class="form-group"><label>spread2</label><input type="number" step="0.01" id="pk-spr2" value="0.2664"></div>
                    <div class="form-group"><label>D2</label><input type="number" step="0.01" id="pk-d2" value="2.3014"></div>
                    <div class="form-group"><label>PPE</label><input type="number" step="0.01" id="pk-ppe" value="0.2846"></div>
                </div>
                <button class="btn-submit" onclick="submitParkinson()">Run Screening</button>
                <div id="res-parkinson" class="result-box"></div>
            </div>

            <!-- 5. Liver Function -->
            <div id="tab-liver" class="panel">
                <div class="panel-title">Liver Function Analysis</div>
                <div class="panel-desc">Evaluate hepatic enzyme levels and blood protein ratios.</div>
                <div class="form-grid">
                    <div class="form-group"><label>Patient Name</label><input type="text" id="lv-name" placeholder="Patient Name"></div>
                    <div class="form-group"><label>Gender</label><select id="lv-sex"><option value="0">Male</option><option value="1">Female</option></select></div>
                    <div class="form-group"><label>Age</label><input type="number" id="lv-age" value="40"></div>
                    <div class="form-group"><label>Total Bilirubin</label><input type="number" step="0.1" id="lv-tb" value="0.9"></div>
                    <div class="form-group"><label>Direct Bilirubin</label><input type="number" step="0.1" id="lv-db" value="0.2"></div>
                    <div class="form-group"><label>Alkaline Phosphatase</label><input type="number" id="lv-alp" value="180"></div>
                    <div class="form-group"><label>Alamine Aminotransferase</label><input type="number" id="lv-alt" value="25"></div>
                    <div class="form-group"><label>Aspartate Aminotransferase</label><input type="number" id="lv-ast" value="30"></div>
                    <div class="form-group"><label>Total Proteins</label><input type="number" step="0.1" id="lv-tp" value="6.8"></div>
                    <div class="form-group"><label>Albumin</label><input type="number" step="0.1" id="lv-alb" value="3.3"></div>
                    <div class="form-group"><label>Albumin/Globulin Ratio</label><input type="number" step="0.1" id="lv-agr" value="0.9"></div>
                </div>
                <button class="btn-submit" onclick="submitLiver()">Analyze Liver Function</button>
                <div id="res-liver" class="result-box"></div>
            </div>

            <!-- 6. Hepatitis Screening -->
            <div id="tab-hepatitis" class="panel">
                <div class="panel-title">Hepatitis Screening</div>
                <div class="panel-desc">Assess liver enzymes, protein markers, and GGT values.</div>
                <div class="form-grid">
                    <div class="form-group"><label>Patient Name</label><input type="text" id="hp-name" placeholder="Patient Name"></div>
                    <div class="form-group"><label>Age</label><input type="number" id="hp-age" value="38"></div>
                    <div class="form-group"><label>Gender</label><select id="hp-sex"><option value="1">Male</option><option value="2">Female</option></select></div>
                    <div class="form-group"><label>Total Bilirubin</label><input type="number" step="0.1" id="hp-alb" value="38.5"></div>
                    <div class="form-group"><label>Direct Bilirubin</label><input type="number" step="0.1" id="hp-alp" value="52.5"></div>
                    <div class="form-group"><label>Alkaline Phosphatase</label><input type="number" step="0.1" id="hp-alt" value="7.7"></div>
                    <div class="form-group"><label>Alamine Aminotransferase</label><input type="number" step="0.1" id="hp-ast" value="22.1"></div>
                    <div class="form-group"><label>Aspartate Aminotransferase</label><input type="number" step="0.1" id="hp-bil" value="7.5"></div>
                    <div class="form-group"><label>Total Proteins</label><input type="number" step="0.1" id="hp-che" value="6.9"></div>
                    <div class="form-group"><label>Albumin</label><input type="number" step="0.1" id="hp-chol" value="3.2"></div>
                    <div class="form-group"><label>Albumin/Globulin Ratio</label><input type="number" step="0.1" id="hp-crea" value="70.0"></div>
                    <div class="form-group"><label>GGT Value</label><input type="number" step="0.1" id="hp-ggt" value="12.1"></div>
                    <div class="form-group"><label>PROT Value</label><input type="number" step="0.1" id="hp-prot" value="69.0"></div>
                </div>
                <button class="btn-submit" onclick="submitHepatitis()">Predict Hepatitis Risk</button>
                <div id="res-hepatitis" class="result-box"></div>
            </div>

            <!-- 7. Pulmonary Oncology -->
            <div id="tab-lung" class="panel">
                <div class="panel-title">Pulmonary Oncology Check</div>
                <div class="panel-desc">Evaluate pulmonary risk factors and respiratory symptoms.</div>
                <div class="form-grid">
                    <div class="form-group"><label>Patient Name</label><input type="text" id="lg-name" placeholder="Patient Name"></div>
                    <div class="form-group"><label>Gender</label><select id="lg-sex"><option value="Male">Male</option><option value="Female">Female</option></select></div>
                    <div class="form-group"><label>Age</label><input type="number" id="lg-age" value="55"></div>
                    <div class="form-group"><label>Smoking</label><select id="lg-smk"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Yellow Fingers</label><select id="lg-yf"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Anxiety</label><select id="lg-anx"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Peer Pressure</label><select id="lg-pp"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Chronic Disease</label><select id="lg-cd"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Fatigue</label><select id="lg-ftg"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Allergy</label><select id="lg-alg"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Wheezing</label><select id="lg-whz"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Alcohol Consuming</label><select id="lg-alc"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Coughing</label><select id="lg-cgh"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Shortness of Breath</label><select id="lg-sob"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Swallowing Difficulty</label><select id="lg-swd"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                    <div class="form-group"><label>Chest Pain</label><select id="lg-cp"><option value="NO">NO</option><option value="YES">YES</option></select></div>
                </div>
                <button class="btn-submit" onclick="submitLung()">Run Pulmonary Screening</button>
                <div id="res-lung" class="result-box"></div>
            </div>

            <!-- 8. Kidney Disease -->
            <div id="tab-kidney" class="panel">
                <div class="panel-title">Chronic Kidney Disease Monitor</div>
                <div class="panel-desc">Assess kidney function indicators, blood chemistry, and urinalysis values.</div>
                <div class="form-grid">
                    <div class="form-group"><label>Patient Name</label><input type="text" id="kd-name" placeholder="Patient Name"></div>
                    <div class="form-group"><label>Age</label><input type="number" id="kd-age" value="48"></div>
                    <div class="form-group"><label>Blood Pressure</label><input type="number" id="kd-bp" value="80"></div>
                    <div class="form-group"><label>Specific Gravity</label><input type="number" step="0.005" id="kd-sg" value="1.020"></div>
                    <div class="form-group"><label>Albumin (0-5)</label><input type="number" id="kd-al" value="1"></div>
                    <div class="form-group"><label>Sugar (0-5)</label><input type="number" id="kd-su" value="0"></div>
                    <div class="form-group"><label>Red Blood Cells</label><select id="kd-rbc"><option value="1">Normal</option><option value="0">Abnormal</option></select></div>
                    <div class="form-group"><label>Pus Cells</label><select id="kd-pc"><option value="1">Normal</option><option value="0">Abnormal</option></select></div>
                    <div class="form-group"><label>Pus Cell Clumps</label><select id="kd-pcc"><option value="0">Not Present</option><option value="1">Present</option></select></div>
                    <div class="form-group"><label>Bacteria</label><select id="kd-ba"><option value="0">Not Present</option><option value="1">Present</option></select></div>
                    <div class="form-group"><label>Blood Glucose Random</label><input type="number" id="kd-bgr" value="121"></div>
                    <div class="form-group"><label>Blood Urea</label><input type="number" id="kd-bu" value="36"></div>
                    <div class="form-group"><label>Serum Creatinine</label><input type="number" step="0.1" id="kd-sc" value="1.2"></div>
                    <div class="form-group"><label>Sodium</label><input type="number" id="kd-sod" value="137"></div>
                    <div class="form-group"><label>Potassium</label><input type="number" step="0.1" id="kd-pot" value="4.4"></div>
                    <div class="form-group"><label>Hemoglobin</label><input type="number" step="0.1" id="kd-hemo" value="15.4"></div>
                    <div class="form-group"><label>Packed Cell Volume</label><input type="number" id="kd-pcv" value="44"></div>
                    <div class="form-group"><label>White Blood Cell Count</label><input type="number" id="kd-wc" value="7800"></div>
                    <div class="form-group"><label>Red Blood Cell Count</label><input type="number" step="0.1" id="kd-rc" value="5.2"></div>
                    <div class="form-group"><label>Hypertension</label><select id="kd-htn"><option value="1">Yes</option><option value="0">No</option></select></div>
                    <div class="form-group"><label>Diabetes Mellitus</label><select id="kd-dm"><option value="1">Yes</option><option value="0">No</option></select></div>
                    <div class="form-group"><label>Coronary Artery Disease</label><select id="kd-cad"><option value="0">No</option><option value="1">Yes</option></select></div>
                    <div class="form-group"><label>Appetite</label><select id="kd-appet"><option value="1">Good</option><option value="0">Poor</option></select></div>
                    <div class="form-group"><label>Pedal Edema</label><select id="kd-pe"><option value="0">No</option><option value="1">Yes</option></select></div>
                    <div class="form-group"><label>Anemia</label><select id="kd-ane"><option value="0">No</option><option value="1">Yes</option></select></div>
                </div>
                <button class="btn-submit" onclick="submitKidney()">Run Kidney Monitor</button>
                <div id="res-kidney" class="result-box"></div>
            </div>
        </main>
    </div>

    <script>
        const DEFAULT_SYMPTOMS = ["abdominal_pain", "abnormal_menstruation", "acidity", "acute_liver_failure", "altered_sensorium", "anxiety", "back_pain", "belly_pain", "blackheads", "bladder_discomfort", "blister", "blood_in_sputum", "bloody_stool", "blurred_and_distorted_vision", "breathlessness", "brittle_nails", "bruising", "burning_micturition", "chest_pain", "chills", "cold_hands_and_feets", "coma", "congestion", "constipation", "continuous_feel_of_urine", "continuous_sneezing", "cough", "cramps", "dark_urine", "dehydration", "depression", "diarrhoea", "dischromic _patches", "distention_of_abdomen", "dizziness", "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger", "extra_marital_contacts", "family_history", "fast_heart_rate", "fatigue", "fluid_overload", "foul_smell_of urine", "headache", "high_fever", "hip_joint_pain", "history_of_alcohol_consumption", "increased_appetite", "indigestion", "inflammatory_nails", "internal_itching", "irregular_sugar_level", "irritability", "irritation_in_anus", "itching", "joint_pain", "knee_pain", "lack_of_concentration", "lethargy", "loss_of_appetite", "loss_of_balance", "loss_of_smell", "loss_of_taste", "malaise", "mild_fever", "mood_swings", "movement_stiffness", "mucoid_sputum", "muscle_pain", "muscle_wasting", "muscle_weakness", "nausea", "neck_pain", "nodal_skin_eruptions", "obesity", "pain_behind_the_eyes", "pain_during_bowel_movements", "pain_in_anal_region", "painful_walking", "palpitations", "passage_of_gases", "patches_in_throat", "phlegm", "polyuria", "prominent_veins_on_calf", "puffy_face_and_eyes", "pus_filled_pimples", "receiving_blood_transfusion", "receiving_unsterile_injections", "red_sore_around_nose", "red_spots_over_body", "redness_of_eyes", "restlessness", "runny_nose", "rusty_sputum", "scurring", "shivering", "silver_like_dusting", "sinus_pressure", "skin_peeling", "skin_rash", "slurred_speech", "small_dents_in_nails", "spinning_movements", "spotting_ urination", "stiff_neck", "stomach_bleeding", "stomach_pain", "sunken_eyes", "sweating", "swelled_lymph_nodes", "swelling_joints", "swelling_of_stomach", "swollen_blood_vessels", "swollen_extremeties", "swollen_legs", "throat_irritation", "tiredness", "toxic_look_(typhos)", "ulcers_on_tongue", "unsteadiness", "visual_disturbances", "vomiting", "watering_from_eyes", "weakness_in_limbs", "weakness_of_one_body_side", "weight_gain", "weight_loss", "yellow_crust_ooze", "yellow_urine", "yellowing_of_eyes", "yellowish_skin"];

        let allSymptoms = DEFAULT_SYMPTOMS;
        let selectedSymptomsSet = new Set();
        let currentFocusIndex = -1;

        function showTab(tabId) {
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }

        function formatSymptomName(sym) {
            if (!sym) return '';
            return sym.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }

        function renderQuickGrid(symptoms) {
            const container = document.getElementById('symptoms-list');
            if (!container) return;
            container.innerHTML = '';
            symptoms.forEach(sym => {
                const label = document.createElement('label');
                label.className = 'symptom-item';
                const isChecked = selectedSymptomsSet.has(sym) ? 'checked' : '';
                label.innerHTML = `<input type="checkbox" value="${sym}" ${isChecked} onchange="toggleSymptom('${sym}', this.checked)"> ${formatSymptomName(sym)}`;
                container.appendChild(label);
            });
        }

        // Render immediately on load
        document.addEventListener('DOMContentLoaded', () => {
            renderQuickGrid(allSymptoms);
        });

        fetch('/api/symptoms')
            .then(res => res.json())
            .then(data => {
                if (data && data.length > 0) {
                    allSymptoms = data;
                    renderQuickGrid(allSymptoms);
                }
            })
            .catch(err => console.error("Using default symptoms list", err));


        function toggleSymptom(sym, checked) {
            if (checked) {
                selectedSymptomsSet.add(sym);
            } else {
                selectedSymptomsSet.delete(sym);
            }
            updateChipsUI();
        }

        function addSymptom(sym) {
            selectedSymptomsSet.add(sym);
            updateChipsUI();
            document.getElementById('symptom-search').value = '';
            closeDropdown();
        }

        function removeSymptom(sym) {
            selectedSymptomsSet.delete(sym);
            updateChipsUI();
        }

        function clearAllSymptoms() {
            selectedSymptomsSet.clear();
            updateChipsUI();
        }

        function updateChipsUI() {
            const container = document.getElementById('selected-chips');
            const countEl = document.getElementById('selected-count');
            countEl.innerText = selectedSymptomsSet.size;

            if (selectedSymptomsSet.size === 0) {
                container.innerHTML = '<span class="empty-chips-msg">No symptoms selected yet. Type above or select from quick list below.</span>';
            } else {
                container.innerHTML = '';
                selectedSymptomsSet.forEach(sym => {
                    const chip = document.createElement('span');
                    chip.className = 'chip';
                    chip.innerHTML = `${formatSymptomName(sym)} <span class="chip-remove" onclick="removeSymptom('${sym}')">&times;</span>`;
                    container.appendChild(chip);
                });
            }

            // Sync quick grid checkboxes
            document.querySelectorAll('#symptoms-list input[type="checkbox"]').forEach(chk => {
                chk.checked = selectedSymptomsSet.has(chk.value);
            });
        }

        function onSearchInput(query) {
            const dropdown = document.getElementById('autocomplete-dropdown');
            const q = query.trim().toLowerCase();
            currentFocusIndex = -1;

            if (!q) {
                dropdown.classList.remove('active');
                dropdown.innerHTML = '';
                return;
            }

            const matches = allSymptoms.filter(sym => {
                const readable = formatSymptomName(sym).toLowerCase();
                return (sym.toLowerCase().includes(q) || readable.includes(q)) && !selectedSymptomsSet.has(sym);
            });

            if (matches.length === 0) {
                dropdown.innerHTML = '<div class="dropdown-item" style="color: var(--text-muted); cursor: default;">No matching unselected symptoms found</div>';
                dropdown.classList.add('active');
                return;
            }

            dropdown.innerHTML = '';
            matches.slice(0, 10).forEach(sym => {
                const item = document.createElement('div');
                item.className = 'dropdown-item';
                item.innerHTML = `<span>${formatSymptomName(sym)}</span> <span class="add-tag">+ Add</span>`;
                item.onclick = () => addSymptom(sym);
                dropdown.appendChild(item);
            });
            dropdown.classList.add('active');
        }

        function closeDropdown() {
            const dropdown = document.getElementById('autocomplete-dropdown');
            dropdown.classList.remove('active');
        }

        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.search-wrapper')) {
                closeDropdown();
            }
        });

        // Keyboard navigation for search input
        document.getElementById('symptom-search').addEventListener('keydown', function(e) {
            const dropdown = document.getElementById('autocomplete-dropdown');
            const items = dropdown.querySelectorAll('.dropdown-item:not([style*="cursor: default"])');
            
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                currentFocusIndex = (currentFocusIndex + 1) % items.length;
                setActiveItem(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                currentFocusIndex = (currentFocusIndex - 1 + items.length) % items.length;
                setActiveItem(items);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (currentFocusIndex > -1 && items[currentFocusIndex]) {
                    items[currentFocusIndex].click();
                } else if (items.length > 0) {
                    items[0].click();
                }
            } else if (e.key === 'Escape') {
                closeDropdown();
            }
        });

        function setActiveItem(items) {
            items.forEach(i => i.classList.remove('active'));
            if (currentFocusIndex >= 0 && items[currentFocusIndex]) {
                items[currentFocusIndex].classList.add('active');
                items[currentFocusIndex].scrollIntoView({ block: 'nearest' });
            }
        }

        function submitSymptomAnalysis() {
            const btn = document.querySelector('#tab-symptom .btn-submit');
            const origText = btn ? btn.innerText : 'Analyze Symptoms';
            
            const selected = Array.from(selectedSymptomsSet);
            if (selected.length === 0) { 
                alert('Please select or search for at least one symptom before running analysis.'); 
                return; 
            }
            
            if (btn) { btn.innerText = 'Analyzing Symptoms...'; btn.disabled = true; }

            fetch('/api/predict/symptom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symptoms: selected })
            })
            .then(res => {
                if (!res.ok) throw new Error('Prediction API server error (' + res.status + ')');
                return res.json();
            })
            .then(data => {
                if (btn) { btn.innerText = origText; btn.disabled = false; }
                const box = document.getElementById('res-symptom');
                box.classList.add('active');
                
                const confidence = data.probability ? (data.probability * 100).toFixed(1) + '%' : 'N/A';
                document.getElementById('res-symptom-header').innerHTML = `<span style="color: var(--primary)">Potential Condition:</span> ${data.prediction} <span style="font-size: 0.85rem; background: rgba(56,189,248,0.2); padding: 0.2rem 0.6rem; border-radius: 6px; margin-left: 0.5rem; color: var(--primary); font-weight: 600;">${confidence} Confidence</span>`;
                document.getElementById('res-symptom-desc').innerText = data.description || 'No description available.';
                
                const precList = document.getElementById('res-symptom-precautions');
                if (data.precautions && Array.isArray(data.precautions)) {
                    precList.innerHTML = '<strong>Recommended Precautions:</strong>' + data.precautions.map(p => `<li>${p}</li>`).join('');
                } else {
                    precList.innerHTML = '';
                }
                
                box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            })
            .catch(err => {
                if (btn) { btn.innerText = origText; btn.disabled = false; }
                alert('Analysis Error: ' + err.message);
            });
        }


        function showResult(boxId, text, isPositive) {
            const box = document.getElementById(boxId);
            box.classList.add('active');
            const color = isPositive ? 'var(--warning)' : 'var(--success)';
            box.innerHTML = `<div class="result-header" style="color: ${color}">${text}</div>`;
            box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }


        function submitDiabetes() {
            const payload = {
                Name: document.getElementById('db-name').value || 'Patient',
                Pregnancies: parseFloat(document.getElementById('db-preg').value),
                Glucose: parseFloat(document.getElementById('db-glu').value),
                BloodPressure: parseFloat(document.getElementById('db-bp').value),
                SkinThickness: parseFloat(document.getElementById('db-skin').value),
                Insulin: parseFloat(document.getElementById('db-ins').value),
                BMI: parseFloat(document.getElementById('db-bmi').value),
                DiabetesPedigreeFunction: parseFloat(document.getElementById('db-ped').value),
                Age: parseFloat(document.getElementById('db-age').value)
            };
            fetch('/api/predict/diabetes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                .then(r => r.json())
                .then(d => showResult('res-diabetes', `${d.name}: ${d.result}`, d.prediction === 1));
        }

        function submitHeart() {
            const payload = {
                name: document.getElementById('ht-name').value || 'Patient',
                age: parseFloat(document.getElementById('ht-age').value),
                sex: parseInt(document.getElementById('ht-sex').value),
                cp: parseInt(document.getElementById('ht-cp').value),
                trestbps: parseFloat(document.getElementById('ht-trestbps').value),
                chol: parseFloat(document.getElementById('ht-chol').value),
                fbs: parseInt(document.getElementById('ht-fbs').value),
                restecg: parseInt(document.getElementById('ht-restecg').value),
                thalach: parseFloat(document.getElementById('ht-thalach').value),
                exang: parseInt(document.getElementById('ht-exang').value),
                oldpeak: parseFloat(document.getElementById('ht-oldpeak').value),
                slope: parseInt(document.getElementById('ht-slope').value),
                ca: parseInt(document.getElementById('ht-ca').value),
                thal: parseInt(document.getElementById('ht-thal').value)
            };
            fetch('/api/predict/heart', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                .then(r => r.json())
                .then(d => showResult('res-heart', `${d.name}: ${d.result}`, d.prediction === 1));
        }

        function submitParkinson() {
            const keys = ['fo','fhi','flo','jit','jitabs','rap','ppq','ddp','shim','shimdb','apq3','apq5','apq','dda','nhr','hnr','rpde','dfa','spr1','spr2','d2','ppe'];
            const payload = { name: document.getElementById('pk-name').value || 'Patient' };
            keys.forEach(k => payload[k] = parseFloat(document.getElementById('pk-' + k).value));
            fetch('/api/predict/parkinson', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                .then(r => r.json())
                .then(d => showResult('res-parkinson', `${d.name}: ${d.result}`, d.prediction === 1));
        }

        function submitLiver() {
            const payload = {
                name: document.getElementById('lv-name').value || 'Patient',
                sex: parseInt(document.getElementById('lv-sex').value),
                age: parseFloat(document.getElementById('lv-age').value),
                tb: parseFloat(document.getElementById('lv-tb').value),
                db: parseFloat(document.getElementById('lv-db').value),
                alp: parseFloat(document.getElementById('lv-alp').value),
                alt: parseFloat(document.getElementById('lv-alt').value),
                ast: parseFloat(document.getElementById('lv-ast').value),
                tp: parseFloat(document.getElementById('lv-tp').value),
                alb: parseFloat(document.getElementById('lv-alb').value),
                agr: parseFloat(document.getElementById('lv-agr').value)
            };
            fetch('/api/predict/liver', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                .then(r => r.json())
                .then(d => showResult('res-liver', `${d.name}: ${d.result}`, d.prediction === 1));
        }

        function submitHepatitis() {
            const payload = {
                name: document.getElementById('hp-name').value || 'Patient',
                age: parseFloat(document.getElementById('hp-age').value),
                sex: parseInt(document.getElementById('hp-sex').value),
                alb: parseFloat(document.getElementById('hp-alb').value),
                alp: parseFloat(document.getElementById('hp-alp').value),
                alt: parseFloat(document.getElementById('hp-alt').value),
                ast: parseFloat(document.getElementById('hp-ast').value),
                bil: parseFloat(document.getElementById('hp-bil').value),
                che: parseFloat(document.getElementById('hp-che').value),
                chol: parseFloat(document.getElementById('hp-chol').value),
                crea: parseFloat(document.getElementById('hp-crea').value),
                ggt: parseFloat(document.getElementById('hp-ggt').value),
                prot: parseFloat(document.getElementById('hp-prot').value)
            };
            fetch('/api/predict/hepatitis', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                .then(r => r.json())
                .then(d => showResult('res-hepatitis', `${d.name}: ${d.result}`, d.prediction === 1));
        }

        function submitLung() {
            const payload = {
                name: document.getElementById('lg-name').value || 'Patient',
                gender: document.getElementById('lg-sex').value,
                age: parseFloat(document.getElementById('lg-age').value),
                smoking: document.getElementById('lg-smk').value,
                yellow_fingers: document.getElementById('lg-yf').value,
                anxiety: document.getElementById('lg-anx').value,
                peer_pressure: document.getElementById('lg-pp').value,
                chronic_disease: document.getElementById('lg-cd').value,
                fatigue: document.getElementById('lg-ftg').value,
                allergy: document.getElementById('lg-alg').value,
                wheezing: document.getElementById('lg-whz').value,
                alcohol_consuming: document.getElementById('lg-alc').value,
                coughing: document.getElementById('lg-cgh').value,
                shortness_of_breath: document.getElementById('lg-sob').value,
                swallowing_difficulty: document.getElementById('lg-swd').value,
                chest_pain: document.getElementById('lg-cp').value
            };
            fetch('/api/predict/lung', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                .then(r => r.json())
                .then(d => showResult('res-lung', `${d.name}: ${d.result}`, d.prediction === 'YES'));
        }

        function submitKidney() {
            const keys = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'];
            const payload = { name: document.getElementById('kd-name').value || 'Patient' };
            keys.forEach(k => payload[k] = parseFloat(document.getElementById('kd-' + k).value));
            fetch('/api/predict/kidney', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                .then(r => r.json())
                .then(d => showResult('res-kidney', `${d.name}: ${d.result}`, d.prediction === 1));
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    m = get_models()
    return jsonify(list(m['disease_model'].all_symptoms))

@app.route('/api/predict/symptom', methods=['POST'])
def predict_symptom():
    data = request.json or {}
    symptoms = data.get('symptoms', [])
    m = get_models()
    X = prepare_symptoms_array(symptoms)
    prediction, prob = m['disease_model'].predict(X)
    description = m['disease_model'].describe_predicted_disease()
    precautions = m['disease_model'].predicted_disease_precautions()
    return jsonify({
        'prediction': prediction,
        'probability': prob,
        'description': description,
        'precautions': precautions
    })

@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json or {}
    m = get_models()
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
    pred = int(m['diabetes'].predict(features)[0])
    result = "Assessment indicates potential metabolic disorder." if pred == 1 else "Assessment shows no metabolic disorder detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart():
    data = request.json or {}
    m = get_models()
    name = data.get('name', 'Patient')
    features = [[
        data.get('age', 0), data.get('sex', 0), data.get('cp', 0),
        data.get('trestbps', 0), data.get('chol', 0), data.get('fbs', 0),
        data.get('restecg', 0), data.get('thalach', 0), data.get('exang', 0),
        data.get('oldpeak', 0), data.get('slope', 0), data.get('ca', 0),
        data.get('thal', 0)
    ]]
    pred = int(m['heart'].predict(features)[0])
    result = "Evaluation suggests cardiovascular risk present." if pred == 1 else "Evaluation indicates no cardiovascular risk detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

@app.route('/api/predict/parkinson', methods=['POST'])
def predict_parkinson():
    data = request.json or {}
    m = get_models()
    name = data.get('name', 'Patient')
    features = [[
        data.get('fo',0), data.get('fhi',0), data.get('flo',0), data.get('jit',0),
        data.get('jitabs',0), data.get('rap',0), data.get('ppq',0), data.get('ddp',0),
        data.get('shim',0), data.get('shimdb',0), data.get('apq3',0), data.get('apq5',0),
        data.get('apq',0), data.get('dda',0), data.get('nhr',0), data.get('hnr',0),
        data.get('rpde',0), data.get('dfa',0), data.get('spr1',0), data.get('spr2',0),
        data.get('d2',0), data.get('ppe',0)
    ]]
    pred = int(m['parkinson'].predict(features)[0])
    result = "Screening indicates potential movement disorder." if pred == 1 else "Screening shows no movement disorder detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

@app.route('/api/predict/liver', methods=['POST'])
def predict_liver():
    data = request.json or {}
    m = get_models()
    name = data.get('name', 'Patient')
    features = [[
        data.get('sex', 0), data.get('age', 0), data.get('tb', 0),
        data.get('db', 0), data.get('alp', 0), data.get('alt', 0),
        data.get('ast', 0), data.get('tp', 0), data.get('alb', 0),
        data.get('agr', 0)
    ]]
    pred = int(m['liver'].predict(features)[0])
    result = "Analysis indicates potential liver dysfunction." if pred == 1 else "Analysis shows normal liver function."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

@app.route('/api/predict/hepatitis', methods=['POST'])
def predict_hepatitis():
    data = request.json or {}
    m = get_models()
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
    pred = int(m['hepatitis'].predict(df)[0])
    result = "Screening indicates potential hepatitis risk." if pred == 1 else "Screening shows no hepatitis detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

@app.route('/api/predict/lung', methods=['POST'])
def predict_lung():
    data = request.json or {}
    m = get_models()
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
    pred = str(m['lung_cancer'].predict(df)[0])
    result = "Screening indicates potential pulmonary risk." if pred == 'YES' else "Screening shows no pulmonary risk detected."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

@app.route('/api/predict/kidney', methods=['POST'])
def predict_kidney():
    data = request.json or {}
    m = get_models()
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
    pred = int(m['chronic'].predict(df)[0])
    result = "Monitoring indicates potential kidney dysfunction." if pred == 1 else "Monitoring shows normal kidney function."
    return jsonify({'name': name, 'prediction': pred, 'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
