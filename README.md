# 🩺 Diagno - AI Multi-Disease Diagnostic System

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

An AI-powered web application for predicting multiple diseases using machine learning models (XGBoost, Random Forest, SVM, Logistic Regression). The system analyzes symptoms, physiological metrics, and lab tests to provide comprehensive risk assessments.

---

## 🚀 Live Hosting Options

### Option 1: Deploy on Vercel (Serverless Web App)

This repository is pre-configured for 1-click deployment on **Vercel** using Python Serverless Functions (`api/index.py`).

#### Step-by-Step Vercel Deployment:

1. **Push your code to GitHub** (if not already done).
2. Go to [Vercel](https://vercel.com/) and log in with your GitHub account.
3. Click **"Add New..."** > **"Project"**.
4. Select your **`Multiple-Disease-Prediction`** GitHub repository and click **Import**.
5. Keep default settings (Vercel auto-detects `vercel.json` and `api/index.py`).
6. Click **Deploy**.
7. Once deployment finishes, copy your live Vercel URL (e.g., `https://your-project-name.vercel.app`).

#### Add Live Link to GitHub:
- On your GitHub repo page, click the ⚙️ **Settings icon** next to **About** (top right of repo home).
- Paste your Vercel URL in the **Website** field and check **Use label "Vercel"** or save changes.

---

### Option 2: Deploy on Streamlit Community Cloud (Native Streamlit)

If you prefer hosting the original Streamlit frontend (`Frontend/app.py`):

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set **Main file path** to `Frontend/app.py`.
4. Click **Deploy!**.

---

## 💻 Local Installation & Setup

### 1. Create Virtual Environment (Python 3.10)

**Windows:**
```powershell
cd Frontend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
cd Frontend
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements_fixed.txt
```

### 3. Launch Applications Locally

- **Streamlit App:**
  ```bash
  streamlit run app.py
  ```
- **Vercel Serverless App (Flask):**
  ```bash
  python ../api/index.py
  ```

---

## ✨ Features & Disease Diagnostic Modules

- **🔍 Symptom Analysis** - AI disease prediction based on 130+ symptoms (XGBoost)
- **🩸 Metabolic Disorder Check** - Diabetes risk evaluation (SVM)
- **❤️ Cardiovascular Risk Assessment** - Heart disease evaluation (Logistic Regression)
- **🧠 Movement Disorder Evaluation** - Parkinson's disease prediction (SVM)
- **🧪 Liver Function Test** - Liver disease assessment (Logistic Regression)
- **🔬 Hepatitis Screening** - Hepatitis risk evaluation (Random Forest)
- **🫁 Pulmonary Oncology Check** - Lung cancer risk assessment (Pipeline Stacking)
- **💧 Kidney Disease Monitor** - Chronic kidney disease monitoring (Logistic Regression)

---

## 🛠️ Technology Stack

- **Backend / Serverless**: Python 3.10+, Flask, Vercel Serverless Functions
- **Frontend**: Glassmorphic Responsive Web Dashboard & Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib, Pandas, NumPy
