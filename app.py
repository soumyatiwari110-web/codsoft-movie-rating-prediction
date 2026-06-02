# ============================================================
# Movie Rating Prediction - app.py
# Author: Your Name
# Description: Flask Web Application Backend
# ============================================================

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================
MODEL_DIR = 'models'

model        = joblib.load(f'{MODEL_DIR}/model.pkl')
scaler       = joblib.load(f'{MODEL_DIR}/scaler.pkl')
le_dict      = joblib.load(f'{MODEL_DIR}/label_encoders.pkl')
feature_cols = joblib.load(f'{MODEL_DIR}/feature_cols.pkl')

CURRENT_YEAR = 2024

def safe_encode(le, value):
    """Encode a label, return 0 if unseen."""
    classes = list(le.classes_)
    if value in classes:
        return le.transform([value])[0]
    return 0

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        genre    = data.get('genre', 'Drama').split(',')[0].strip()
        director = data.get('director', 'Unknown')
        actor1   = data.get('actor1', 'Unknown')
        actor2   = data.get('actor2', 'Unknown')
        actor3   = data.get('actor3', 'Unknown')
        duration = float(data.get('duration', 120))
        votes    = float(data.get('votes', 1000))
        year     = float(data.get('year', 2010))

        movie_age   = CURRENT_YEAR - year
        log_votes   = np.log1p(votes)
        total_actors = sum([
            1 if actor1 != 'Unknown' else 0,
            1 if actor2 != 'Unknown' else 0,
            1 if actor3 != 'Unknown' else 0
        ])

        # Encode
        genre_enc    = safe_encode(le_dict['Primary_Genre'], genre)
        director_enc = safe_encode(le_dict['Director'],      director)
        a1_enc       = safe_encode(le_dict['Actor 1'],       actor1)
        a2_enc       = safe_encode(le_dict['Actor 2'],       actor2)
        a3_enc       = safe_encode(le_dict['Actor 3'],       actor3)

        # Build feature vector in correct order
        features = np.array([[
            duration, log_votes, movie_age, total_actors,
            genre_enc, director_enc, a1_enc, a2_enc, a3_enc
        ]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        prediction = round(float(prediction), 1)
        prediction = max(1.0, min(10.0, prediction))  # clamp to [1, 10]

        return jsonify({'rating': prediction, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)