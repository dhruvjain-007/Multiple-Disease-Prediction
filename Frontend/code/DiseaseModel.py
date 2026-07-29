import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class DiseaseModel:

    def __init__(self):
        self.all_symptoms = None
        self.symptoms = None
        self.pred_disease = None
        self.model = None
        self.diseases = self.disease_list(os.path.join(BASE_DIR, 'data', 'dataset.csv'))

    def load_xgboost(self, model_path=None):
        self.load_model(model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(BASE_DIR, 'models', 'symptom_model.sav')
        elif not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
        
        if model_path.endswith('.json'):
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = joblib.load(model_path)

    def save_xgboost(self, model_path):
        if self.model and hasattr(self.model, 'save_model'):
            if not os.path.isabs(model_path):
                model_path = os.path.join(BASE_DIR, model_path)
            self.model.save_model(model_path)

    def predict(self, X):
        if self.model is None:
            self.load_model()
        self.symptoms = X
        preds = self.model.predict(self.symptoms)
        self.pred_disease = str(preds[0])
        probs = self.model.predict_proba(self.symptoms)[0]
        disease_probability = float(np.max(probs))
        return self.pred_disease, disease_probability

    def describe_disease(self, disease_name):
        if disease_name not in self.diseases:
            return "That disease is not contemplated in this model"
        desc_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'symptom_Description.csv'))
        desc_df['Disease'] = desc_df['Disease'].astype(str).str.strip()
        matching = desc_df[desc_df['Disease'] == disease_name]['Description']
        if len(matching) > 0 and pd.notna(matching.values[0]):
            return str(matching.values[0]).strip()
        return "No detailed description available."

    def describe_predicted_disease(self):
        if self.pred_disease is None:
            return "No predicted disease yet"
        return self.describe_disease(self.pred_disease)
    
    def disease_precautions(self, disease_name):
        default_prec = ["Consult a doctor", "Get adequate rest", "Stay hydrated", "Follow healthy diet"]
        if disease_name not in self.diseases:
            return default_prec
        prec_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'symptom_precaution.csv'))
        prec_df['Disease'] = prec_df['Disease'].astype(str).str.strip()
        matching = prec_df[prec_df['Disease'] == disease_name].filter(regex='Precaution')
        if len(matching) > 0:
            row = matching.values[0]
            clean_list = [str(item).strip() for item in row if pd.notna(item) and str(item).strip() and str(item).strip().lower() != 'nan']
            if clean_list:
                return clean_list
        return default_prec

    def predicted_disease_precautions(self):
        if self.pred_disease is None:
            return ["Consult a doctor", "Get adequate rest", "Stay hydrated", "Follow healthy diet"]
        return self.disease_precautions(self.pred_disease)


    def disease_list(self, kaggle_dataset):
        df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_dataset.tsv'), sep='\t')
        y_data = df.iloc[:,-1]
        X_data = df.iloc[:,:-1]
        self.all_symptoms = X_data.columns
        y_data = y_data.astype('category')
        return y_data.cat.categories