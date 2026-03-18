import pickle
from preprocessing import clean_text

# Load model & vectorizer
model = pickle.load(open('../models/model.pkl', 'rb'))
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

def predict_job(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]

    return prediction, max(prob)