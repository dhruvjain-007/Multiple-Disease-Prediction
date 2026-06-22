# Multi-Disease Diagnostic System

A Streamlit web application for predicting multiple diseases using machine learning models. The app analyzes symptoms, medical history, and test results to provide disease risk assessments.

## Prerequisites

- **Python 3.10** (required for all dependencies)

## Installation & Setup

### 1. Create Virtual Environment (Python 3.10)

**Windows:**
```powershell
cd Frontend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
cd Frontend
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements_fixed.txt
```

### 3. Launch the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` by default, or at a different port if Streamlit selects one automatically.

## Features

- **Symptom Analysis** - Disease prediction based on symptoms
- **Metabolic Disorder Check** - Diabetes and metabolic health assessment
- **Cardiovascular Risk Assessment** - Heart disease evaluation
- **Movement Disorder Evaluation** - Parkinson's disease prediction
- **Liver Function Test** - Liver disease assessment
- **Hepatitis Screening** - Hepatitis risk evaluation
- **Pulmonary Oncology Check** - Lung cancer prediction
- **Kidney Disease Monitor** - Chronic kidney disease assessment

## Technology Stack

- [Streamlit](https://docs.streamlit.io/) - Frontend framework
- [Python 3.10+](https://www.python.org) - Programming language
- [Scikit-learn](https://scikit-learn.org/) - Machine learning models
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting framework
- [Plotly](https://plotly.com/) - Data visualization



