import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import clean_text

# Load dataset
df = pd.read_csv('data/fake_job_dataset.csv', engine='python', on_bad_lines='skip')

# Assume dataset has 'description' and 'fraudulent'
df['description'] = df['description'].fillna('')
df['cleaned'] = df['description'].apply(clean_text)

X = df['cleaned']
y = df['fraudulent']

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import os
import pickle

os.makedirs('models', exist_ok=True)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)