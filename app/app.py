from flask import Flask, render_template, request
import pickle
import os
import sys

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
# Load model and vectorizer
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None

    if request.method == 'POST':
        job_text = request.form['job_description']

        # Preprocess
        cleaned = clean_text(job_text)

        # Vectorize
        vectorized = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        # Output
        result = "🚨 Fake Job" if prediction == 1 else "✅ Real Job"
        confidence = round(max(prob) * 100, 2)

    return render_template('index.html', result=result, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)