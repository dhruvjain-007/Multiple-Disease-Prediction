import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SYMPTOMS_LIST = ["abdominal_pain", "abnormal_menstruation", "acidity", "acute_liver_failure", "altered_sensorium", "anxiety", "back_pain", "belly_pain", "blackheads", "bladder_discomfort", "blister", "blood_in_sputum", "bloody_stool", "blurred_and_distorted_vision", "breathlessness", "brittle_nails", "bruising", "burning_micturition", "chest_pain", "chills", "cold_hands_and_feets", "coma", "congestion", "constipation", "continuous_feel_of_urine", "continuous_sneezing", "cough", "cramps", "dark_urine", "dehydration", "depression", "diarrhoea", "dischromic _patches", "distention_of_abdomen", "dizziness", "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger", "extra_marital_contacts", "family_history", "fast_heart_rate", "fatigue", "fluid_overload", "foul_smell_of urine", "headache", "high_fever", "hip_joint_pain", "history_of_alcohol_consumption", "increased_appetite", "indigestion", "inflammatory_nails", "internal_itching", "irregular_sugar_level", "irritability", "irritation_in_anus", "itching", "joint_pain", "knee_pain", "lack_of_concentration", "lethargy", "loss_of_appetite", "loss_of_balance", "loss_of_smell", "loss_of_taste", "malaise", "mild_fever", "mood_swings", "movement_stiffness", "mucoid_sputum", "muscle_pain", "muscle_wasting", "muscle_weakness", "nausea", "neck_pain", "nodal_skin_eruptions", "obesity", "pain_behind_the_eyes", "pain_during_bowel_movements", "pain_in_anal_region", "painful_walking", "palpitations", "passage_of_gases", "patches_in_throat", "phlegm", "polyuria", "prominent_veins_on_calf", "puffy_face_and_eyes", "pus_filled_pimples", "receiving_blood_transfusion", "receiving_unsterile_injections", "red_sore_around_nose", "red_spots_over_body", "redness_of_eyes", "restlessness", "runny_nose", "rusty_sputum", "scurring", "shivering", "silver_like_dusting", "sinus_pressure", "skin_peeling", "skin_rash", "slurred_speech", "small_dents_in_nails", "spinning_movements", "spotting_ urination", "stiff_neck", "stomach_bleeding", "stomach_pain", "sunken_eyes", "sweating", "swelled_lymph_nodes", "swelling_joints", "swelling_of_stomach", "swollen_blood_vessels", "swollen_extremeties", "swollen_legs", "throat_irritation", "tiredness", "toxic_look_(typhos)", "ulcers_on_tongue", "unsteadiness", "visual_disturbances", "vomiting", "watering_from_eyes", "weakness_in_limbs", "weakness_of_one_body_side", "weight_gain", "weight_loss", "yellow_crust_ooze", "yellow_urine", "yellowing_of_eyes", "yellowish_skin"]

DISEASE_LIST = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Covid', 'Dengue', 'Diabetes', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']

DESCRIPTIONS_MAP = {
  "Fungal infection": "In humans, fungal infections occur when an invading fungus takes over an area of the body and is too much for the immune system to handle. Fungi can live in the air, soil, water, and plants. There are also some fungi that live naturally in the human body. Like many microbes, there are helpful fungi and harmful fungi.",
  "Allergy": "An allergy is an immune system response to a foreign substance that's not typically harmful to your body. These foreign substances are called allergens. They can include certain foods, pollen, or pet dander. Your immune system's job is to keep you healthy by fighting harmful pathogens.",
  "GERD": "Gastroesophageal reflux disease, or GERD, is a digestive disorder that affects the ring of muscle between your esophagus and stomach. This ring is the lower esophageal sphincter (LES). If you have GERD, experience heartburn or acid indigestion.",
  "Chronic cholestasis": "Chronic cholestatic liver diseases result in progressive destruction of bile ducts, cholestasis, biliary cirrhosis, and hepatic failure. The 2 major categories of chronic cholestatic liver disease in adults are primary biliary cirrhosis (PBC) and primary sclerosing cholangitis (PSC).",
  "Drug Reaction": "An adverse drug reaction (ADR) is an injury caused by taking a medication. ADRs may occur following a single dose or prolonged administration of a drug or result from the combination of two or more drugs.",
  "Peptic ulcer diseae": "Peptic ulcers are open sores that develop on the inside lining of your stomach and the upper portion of your small intestine. The most common symptom of a peptic ulcer is stomach pain. Peptic ulcers include: Gastric ulcers that occur on the inside of the stomach.",
  "AIDS": "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). By damaging your immune system, HIV interferes with your body's ability to fight infection and disease.",
  "Diabetes": "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.",
  "Gastroenteritis": "Gastroenteritis is an inflammation of the digestive tract, particularly the stomach, and large and small intestines. Viral and bacterial gastroenteritis are intestinal infections associated with symptoms of diarrhea , abdominal cramps, nausea , and vomiting .",
  "Bronchial Asthma": "Bronchial asthma is a medical condition which causes the airway path of the lungs to swell and narrow. Due to this swelling, the air path produces excess mucus making it hard to breathe, which results in coughing, short breath, and wheezing.",
  "Hypertension": "Hypertension (HTN or HT), also known as high blood pressure (HBP), is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms.",
  "Migraine": "A migraine can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days, and the pain can be so severe that it interferes with your daily activities.",
  "Cervical spondylosis": "Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs).",
  "Paralysis (brain hemorrhage)": "Intracerebral hemorrhage (ICH) is when blood suddenly bursts into brain tissue, causing damage to your brain. Symptoms usually appear suddenly during ICH. They include headache, weakness, confusion, and paralysis, particularly on one side of your body.",
  "Jaundice": "Yellow staining of the skin and sclerae (the whites of the eyes) by abnormally high blood levels of the bile pigment bilirubin. The yellowing extends to other tissues and body fluids.",
  "Malaria": "An infectious disease caused by protozoan parasites that can be transmitted by the bite of the infected female Anopheles mosquito or by contaminated needles or blood transfusions.",
  "Chicken pox": "Chickenpox is a highly contagious disease caused by the varicella-zoster virus (VZV). It causes an itchy, blister-like rash among other symptoms.",
  "Dengue": "an acute infectious disease caused by a flavivirus (species Dengue virus of the genus Flavivirus), transmitted by aedes mosquitoes, and characterized by headache, severe joint pain, and a rash.",
  "Typhoid": "An acute illness associated with fever caused by the Salmonella enterica serotype Typhi bacteria. It can also be caused by Salmonella paratyphi.",
  "hepatitis A": "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. The virus is one of several types of hepatitis viruses that cause inflammation and affect your liver's ability to function.",
  "Hepatitis B": "Hepatitis B is a serious liver infection caused by the hepatitis B virus (HBV). For most people, hepatitis B is short term, also called acute, and lasts less than six months.",
  "Hepatitis C": "Hepatitis C is a viral infection that causes liver inflammation, sometimes leading to serious liver damage. The hepatitis C virus (HCV) spreads through contaminated blood.",
  "Hepatitis D": "Hepatitis D, also known as the hepatitis delta virus, is an infection that causes the liver to become inflamed. This swelling can impair liver function and cause long-term liver problems.",
  "Hepatitis E": "A rare form of liver inflammation caused by infection with the hepatitis E virus (HEV). It is transmitted via food or drink handled by an infected person or through infected water supplies.",
  "Alcoholic hepatitis": "Alcoholic hepatitis is a diseased, inflammatory condition of the liver caused by heavy alcohol consumption over a long period of time.",
  "Tuberculosis": "Tuberculosis (TB) is an infectious disease usually caused by Mycobacterium tuberculosis (MTB) bacteria. Tuberculosis generally affects the lungs, but can also affect other parts of the body.",
  "Common Cold": "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold.",
  "Pneumonia": "Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli.",
  "Dimorphic hemmorhoids(piles)": "Hemorrhoids, also called piles, are swollen veins in your anus and lower rectum, similar to varicose veins. Hemorrhoids can develop inside the rectum (internal hemorrhoids) or under the skin around the anus (external hemorrhoids).",
  "Heart attack": "The death of heart muscle due to the loss of blood supply. The loss of blood supply is usually caused by a complete blockage of a coronary artery, one of the arteries that supplies blood to the heart muscle.",
  "Varicose veins": "Varicose veins are swollen, twisted veins that you can see just under the skin. They usually occur in the legs, but also can form in other parts of the body.",
  "Hypothyroidism": "Hypothyroidism (underactive thyroid) is a condition in which your thyroid gland doesn't produce enough of certain crucial hormones.",
  "Hyperthyroidism": "Hyperthyroidism (overactive thyroid) occurs when your thyroid gland produces too much of the hormone thyroxine.",
  "Hypoglycemia": "Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. Glucose is your body's main energy source.",
  "Osteoarthristis": "Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage that cushions the ends of your bones wears down over time.",
  "Arthritis": "Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of arthritis are joint pain and stiffness.",
  "(vertigo) Paroymsal  Positional Vertigo": "Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — the sudden sensation that you're spinning or that the inside of your head is spinning.",
  "Acne": "Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells. It causes whiteheads, blackheads or pimples.",
  "Urinary tract infection": "An infection in any part of the urinary system, the kidneys, bladder or urethra. Urinary tract infections are more common in women.",
  "Psoriasis": "Psoriasis is a skin disease that causes red, itchy scaly patches, most commonly on the knees, elbows, trunk and scalp.",
  "Impetigo": "Impetigo is a common and contagious skin infection that mainly affects infants and children. It usually appears as reddish sores on the face.",
  "Covid": "COVID-19 affects different people in different ways. Most infected people will develop mild to moderate respiratory illness and recover without requiring special treatment."
}

PRECAUTIONS_MAP = {
  "Drug Reaction": ["stop irritation", "consult nearest hospital", "stop taking drug", "follow up"],
  "Malaria": ["Consult nearest hospital", "avoid oily food", "avoid non veg food", "keep mosquitos out"],
  "Allergy": ["apply calamine", "cover area with bandage", "use ice to compress itching"],
  "Hypothyroidism": ["reduce stress", "exercise", "eat healthy", "get proper sleep"],
  "Psoriasis": ["wash hands with warm soapy water", "stop bleeding using pressure", "consult doctor", "salt baths"],
  "GERD": ["avoid fatty spicy food", "avoid lying down after eating", "maintain healthy weight", "exercise"],
  "Chronic cholestasis": ["cold baths", "anti itch medicine", "consult doctor", "eat healthy"],
  "hepatitis A": ["Consult nearest hospital", "wash hands through", "avoid fatty spicy food", "medication"],
  "Osteoarthristis": ["acetaminophen", "consult nearest hospital", "follow up", "salt baths"],
  "(vertigo) Paroymsal  Positional Vertigo": ["lie down", "avoid sudden change in body", "avoid abrupt head movment", "relax"],
  "Hypoglycemia": ["lie down on side", "check in pulse", "drink sugary drinks", "consult doctor"],
  "Acne": ["bath twice", "avoid fatty spicy food", "drink plenty of water", "avoid too many products"],
  "Diabetes": ["have balanced diet", "exercise", "consult doctor", "follow up"],
  "Impetigo": ["soak affected area in warm water", "use antibiotics", "remove scabs with wet compressed cloth", "consult doctor"],
  "Hypertension": ["meditation", "salt baths", "reduce stress", "get proper sleep"],
  "Peptic ulcer diseae": ["avoid fatty spicy food", "consume probiotic food", "eliminate milk", "limit alcohol"],
  "Dimorphic hemmorhoids(piles)": ["avoid fatty spicy food", "consume witch hazel", "warm bath with epsom salt", "consume alovera juice"],
  "Common Cold": ["drink vitamin c rich drinks", "take vapour", "avoid cold food", "keep fever in check"],
  "Chicken pox": ["use neem in bathing", "consume neem leaves", "take vaccine", "avoid public places"],
  "Cervical spondylosis": ["use heating pad or cold pack", "exercise", "take otc pain reliver", "consult doctor"],
  "Hyperthyroidism": ["eat healthy", "massage", "use lemon balm", "take radioactive iodine treatment"],
  "Urinary tract infection": ["drink plenty of water", "increase vitamin c intake", "drink cranberry juice", "take probiotics"],
  "Varicose veins": ["lie down flat and raise the leg high", "use oinments", "use vein compression", "dont stand still for long"],
  "AIDS": ["avoid open cuts", "wear ppe if possible", "consult doctor", "follow up"],
  "Paralysis (brain hemorrhage)": ["massage", "eat healthy", "exercise", "consult doctor"],
  "Typhoid": ["eat high calorie vegitables", "antiboitic therapy", "consult doctor", "medication"],
  "Hepatitis B": ["consult nearest hospital", "vaccination", "eat healthy", "medication"],
  "Fungal infection": ["bath twice", "use detol or neem in bathing water", "keep infected area dry", "use clean cloths"],
  "Hepatitis C": ["Consult nearest hospital", "vaccination", "eat healthy", "medication"],
  "Migraine": ["meditation", "reduce stress", "use poloroid glasses in sun", "consult doctor"],
  "Bronchial Asthma": ["switch to loose cloothing", "take deep breaths", "get away from trigger", "seek help"],
  "Alcoholic hepatitis": ["stop alcohol consumption", "consult doctor", "medication", "follow up"],
  "Jaundice": ["drink plenty of water", "consume milk thistle", "eat fruits and high fiberous food", "medication"],
  "Hepatitis E": ["stop alcohol consumption", "rest", "consult doctor", "medication"],
  "Dengue": ["drink papaya leaf juice", "avoid fatty spicy food", "keep mosquitos away", "keep hydrated"],
  "Hepatitis D": ["consult doctor", "medication", "eat healthy", "follow up"],
  "Heart attack": ["call ambulance", "chew or swallow asprin", "keep calm"],
  "Pneumonia": ["consult doctor", "medication", "rest", "follow up"],
  "Arthritis": ["exercise", "use hot and cold therapy", "try acupuncture", "massage"],
  "Gastroenteritis": ["stop eating solid food for while", "try taking small sips of water", "rest", "ease back into eating"],
  "Tuberculosis": ["cover mouth", "consult doctor", "medication", "rest"],
  "Covid": ["cover mouth", "social distancing", "wear mask", "medication"]
}

class DiseaseModel:

    def __init__(self):
        self.all_symptoms = SYMPTOMS_LIST
        self.symptoms = None
        self.pred_disease = None
        self.model = None
        self.diseases = DISEASE_LIST

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
        return DESCRIPTIONS_MAP.get(str(disease_name).strip(), "No detailed description available.")

    def describe_predicted_disease(self):
        if self.pred_disease is None:
            return "No predicted disease yet"
        return self.describe_disease(self.pred_disease)
    
    def disease_precautions(self, disease_name):
        return PRECAUTIONS_MAP.get(str(disease_name).strip(), ["Consult a doctor", "Get adequate rest", "Stay hydrated", "Follow healthy diet"])

    def predicted_disease_precautions(self):
        if self.pred_disease is None:
            return ["Consult a doctor", "Get adequate rest", "Stay hydrated", "Follow healthy diet"]
        return self.disease_precautions(self.pred_disease)

    def disease_list(self, kaggle_dataset=None):
        return DISEASE_LIST