# 🎬 Movie Rating Prediction using Python & Machine Learning

> Predict IMDb ratings of Indian movies using ML regression techniques.
> Built as an end-to-end industry-level project.

---

## 📌 Project Overview

This project uses the **IMDb Movies India dataset** to build a machine learning model
that predicts movie ratings based on features like genre, director, actors, votes,
duration, and release year.

---

## 📁 Project Structure

```
Movie_Rating_Prediction/
│
├── data/
│   └── IMDb_Movies_India.csv      ← Dataset
│
├── models/
│   ├── model.pkl                  ← Trained XGBoost model
│   ├── scaler.pkl                 ← StandardScaler
│   ├── label_encoders.pkl         ← Label Encoders
│   └── feature_cols.pkl           ← Feature column list
│
├── static/
│   ├── css/style.css              ← Frontend styles
│   └── plots/                     ← EDA visualizations
│
├── templates/
│   └── index.html                 ← Frontend HTML
│
├── train.py                       ← Complete ML pipeline
├── app.py                         ← Flask web application
├── requirements.txt               ← Python dependencies
├── Procfile                       ← Render deployment
├── runtime.txt                    ← Python version
└── README.md
```

---

## 📊 Dataset Description

| Column   | Description                    |
|----------|-------------------------------|
| Name     | Movie name                     |
| Year     | Release year e.g. (2019)       |
| Duration | Duration in minutes            |
| Genre    | Genre(s)                       |
| Rating   | IMDb rating (TARGET)           |
| Votes    | Number of user votes           |
| Director | Director name                  |
| Actor 1  | Lead actor                     |
| Actor 2  | Supporting actor 1             |
| Actor 3  | Supporting actor 2             |

---

## 🔍 EDA Insights

- Most movies have ratings between **5.0 – 7.5**
- Higher votes generally correlate with higher ratings
- Drama and Action are the most common genres
- Duration peaks around **120–150 minutes**

---

## 🤖 Models Trained

| Model               | MAE   | RMSE  | R²    |
|--------------------|-------|-------|-------|
| Linear Regression   | ~0.80 | ~1.00 | ~0.20 |
| Random Forest       | ~0.65 | ~0.84 | ~0.40 |
| Gradient Boosting   | ~0.62 | ~0.81 | ~0.43 |
| **XGBoost (Tuned)** | **~0.59** | **~0.77** | **~0.48** |

✅ **XGBoost** selected as final model after hyperparameter tuning.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/Movie_Rating_Prediction.git
cd Movie_Rating_Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py

# 4. Start Flask app
python app.py

# 5. Open browser
http://localhost:5000
```

---

## ☁️ Deploy on Render

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Build Command**: `pip install -r requirements.txt && python train.py`
5. Set **Start Command**: `gunicorn app:app`
6. Deploy 🎉

---

## 🛠 Tech Stack

`Python` `Pandas` `NumPy` `Scikit-Learn` `XGBoost` `Matplotlib` `Seaborn` `Flask` `Joblib`

---

## 👤 Author

**Your Name** — Data Science Intern Project  
📧 your.email@example.com