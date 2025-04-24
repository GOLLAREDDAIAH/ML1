import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
data = {
    'text': [
        "patient presents with fever, cough, and shortness of breath - suspected respiratory infection",
        "Sharp chest pain radiating to left arm , ECG changes observed",
        "Headache, dizziness, and high blood pressure readings",
        "Abdonimal pain, nausea, and tenderness in right lower quardent",
        "Joint swelling, morning stiffness, and faitgue",
        "Frequent urination, increased thirst, elevated blood glucose levels",
        "Skin rash, itching, and allgeric reaction to mediaction",
        "Difficulty braething, wheezing , and know asthma history"
    ],
    'disease': [
        'Respiratory Infection',
        'Cardiac Issue',
        'Hypertension',
        'Appendicitis',
        'Arthritis',
        'Diabetes',
        'Allergic Reaction',
        'Asthma'
    ]
}

df = pd.DataFrame(data)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Features and labels
X = df['cleaned_text']
y = df['disease']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved.")
 
