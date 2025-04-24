from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from difflib import SequenceMatcher

app = Flask(__name__)

# Symptom-disease map for fuzzy matching
symptom_disease_map = {
    "fever, cough, and shortness of breath": "Respiratory Infection",
    "sharp chest pain radiating to left arm": "Cardiac Issue",
    "headache, dizziness, and high blood pressure": "Hypertension",
    "abdominal pain, nausea, and tenderness in right lower quadrant": "Appendicitis",
    "joint swelling, morning stiffness, and fatigue": "Arthritis",
    "frequent urination, increased thirst, elevated blood glucose": "Diabetes",
    "skin rash, itching, and allergic reaction": "Allergic Reaction",
    "difficulty breathing, wheezing, and known asthma history": "Asthma"
}

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess user input for ML model
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    return text

# Fuzzy similarity function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Fallback function for fuzzy symptom matching
def get_best_match(user_input):
    user_symptoms = [s.strip().lower() for s in user_input.split(',') if s.strip()]
    best_match = None
    max_score = 0

    for symptom_str, disease in symptom_disease_map.items():
        disease_symptoms = [s.strip().lower() for s in symptom_str.split(',')]
        score = 0

        for user_symptom in user_symptoms:
            for disease_symptom in disease_symptoms:
                if similar(user_symptom, disease_symptom) > 0.6:  # Similarity threshold
                    score += 1

        if score > max_score:
            max_score = score
            best_match = disease

    return best_match if max_score > 0 else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '')
    
    if not symptoms:
        return render_template('index.html', error='No symptoms provided')

    # Preprocess and vectorize for ML model
    cleaned = preprocess_text(symptoms)
    vectorized = vectorizer.transform([cleaned])
    
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    max_probability = max(probabilities)

    if max_probability < 0.3:
        # Low confidence → try fuzzy match
        fallback_prediction = get_best_match(symptoms)
        if fallback_prediction:
            return render_template('index.html', symptoms=symptoms, prediction=fallback_prediction)
        else:
            return render_template('index.html', symptoms=symptoms, no_match=True)

    return render_template('index.html', symptoms=symptoms, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
