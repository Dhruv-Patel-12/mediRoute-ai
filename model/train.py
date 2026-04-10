import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Setup absolute paths to ensure the script runs from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'Training.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'Testing.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'features.pkl')

# Map exact Kaggle disease names to your 8 Specialties
SPECIALTY_MAP = {
    'common cold': 'General Physician', 'pneumonia': 'General Physician', 'typhoid': 'General Physician',
    'malaria': 'General Physician', 'dengue': 'General Physician', 'tuberculosis': 'General Physician',
    'drug reaction': 'General Physician', 'urinary tract infection': 'General Physician',
    'heart attack': 'Cardiologist', 'hypertension': 'Cardiologist', 'hypertension ': 'Cardiologist',
    'acne': 'Dermatologist', 'psoriasis': 'Dermatologist', 'fungal infection': 'Dermatologist',
    'chicken pox': 'Dermatologist', 'impetigo': 'Dermatologist',
    'migraine': 'Neurologist', 'paralysis (brain hemorrhage)': 'Neurologist', 
    'cervical spondylosis': 'Neurologist', 'epilepsy': 'Neurologist',
    '(vertigo) paroymsal  positional vertigo': 'ENT Specialist',
    'arthritis': 'Orthopedic', 'osteoarthristis': 'Orthopedic', 'osteoarthritis': 'Orthopedic',
    'gerd': 'Gastroenterologist', 'peptic ulcer diseae': 'Gastroenterologist', 
    'gastroenteritis': 'Gastroenterologist', 'jaundice': 'Gastroenterologist', 
    'hepatitis a': 'Gastroenterologist', 'hepatitis b': 'Gastroenterologist', 
    'hepatitis c': 'Gastroenterologist', 'hepatitis d': 'Gastroenterologist', 
    'hepatitis e': 'Gastroenterologist', 'alcoholic hepatitis': 'Gastroenterologist',
    'allergy': 'ENT Specialist', 'sinusitis': 'ENT Specialist',
    'depression': 'Psychiatrist', 'anxiety': 'Psychiatrist'
}

def map_specialty(disease):
    disease_lower = str(disease).lower().strip()
    # Default to General Physician if the disease isn't explicitly in the map
    return SPECIALTY_MAP.get(disease_lower, 'General Physician')

def train_model():
    print("Loading Training and Testing datasets...")
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError("Missing Training.csv or Testing.csv in the data/ folder.")
    
    # Load the datasets
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # The Kaggle CSV files usually have trailing commas which create a completely empty junk column at the end
    # This automatically drops any columns that are entirely empty (NaN)
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')
    
    print("Mapping Kaggle diseases to Doctor Specialties...")
    y_train = train_df['prognosis'].apply(map_specialty)
    y_test = test_df['prognosis'].apply(map_specialty)
    
    # Drop the prognosis (target) column to isolate our 132 symptom features
    X_train = train_df.drop(columns=['prognosis'])
    X_test = test_df.drop(columns=['prognosis'])
    
    # Ensure Testing and Training columns are exactly aligned
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    feature_names = list(X_train.columns)
    
    print("Training Random Forest Classifier on Training.csv...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    print("Evaluating model on Testing.csv...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and feature configuration...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    
    print("Done! Both model.pkl and features.pkl saved successfully.")

if __name__ == "__main__":
    train_model()