# ============================================================
# Movie Rating Prediction - train.py
# Author: Your Name
# Description: Complete ML pipeline - EDA, Cleaning, Modeling
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = 'data/IMDb Movies India.csv'
MODELS_DIR  = 'models'
PLOTS_DIR   = 'static/plots'
CURRENT_YEAR = 2024

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ============================================================
# PHASE 1 — LOAD DATA
# ============================================================
print("\n" + "="*60)
print("PHASE 1: LOADING DATASET")
print("="*60)

df = pd.read_csv(DATA_PATH, encoding='latin1')

print(f"Shape      : {df.shape}")
print(f"Columns    : {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nSample Rows:\n{df.head()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# ============================================================
# PHASE 2 — DATA CLEANING
# ============================================================
print("\n" + "="*60)
print("PHASE 2: DATA CLEANING")
print("="*60)

# --- Year: "(2019)" → 2019 ---
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# --- Duration: "109 min" → 109 ---
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# --- Votes: "1,234" → 1234 ---
df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# --- Drop rows where Rating (target) is missing ---
df.dropna(subset=['Rating'], inplace=True)

# --- Drop duplicates ---
df.drop_duplicates(inplace=True)

# --- Fill remaining numeric NaNs with median ---
for col in ['Duration', 'Votes', 'Year']:
    df[col].fillna(df[col].median(), inplace=True)

# --- Fill remaining categorical NaNs with 'Unknown' ---
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col].fillna('Unknown', inplace=True)

print(f"Cleaned Shape: {df.shape}")
print(f"Missing after cleaning:\n{df.isnull().sum()}")

# ============================================================
# PHASE 3 — EDA (saves plots to static/plots/)
# ============================================================
print("\n" + "="*60)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("="*60)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# 1. Rating Distribution
fig, ax = plt.subplots()
sns.histplot(df['Rating'], bins=30, kde=True, color='steelblue', ax=ax)
ax.set_title('Rating Distribution')
ax.set_xlabel('Rating')
fig.savefig(f'{PLOTS_DIR}/rating_distribution.png', bbox_inches='tight')
plt.close()

# 2. Duration Distribution
fig, ax = plt.subplots()
sns.histplot(df['Duration'].dropna(), bins=30, kde=True, color='coral', ax=ax)
ax.set_title('Duration Distribution')
ax.set_xlabel('Duration (minutes)')
fig.savefig(f'{PLOTS_DIR}/duration_distribution.png', bbox_inches='tight')
plt.close()

# 3. Votes Distribution (log scale)
fig, ax = plt.subplots()
sns.histplot(np.log1p(df['Votes'].dropna()), bins=30, kde=True, color='green', ax=ax)
ax.set_title('Votes Distribution (log scale)')
ax.set_xlabel('log(Votes)')
fig.savefig(f'{PLOTS_DIR}/votes_distribution.png', bbox_inches='tight')
plt.close()

# 4. Year Distribution
fig, ax = plt.subplots()
sns.histplot(df['Year'].dropna(), bins=40, kde=True, color='purple', ax=ax)
ax.set_title('Year Distribution')
fig.savefig(f'{PLOTS_DIR}/year_distribution.png', bbox_inches='tight')
plt.close()

# 5. Top 10 Genres
fig, ax = plt.subplots()
genre_counts = df['Genre'].str.split(',').explode().str.strip().value_counts().head(10)
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis', ax=ax)
ax.set_title('Top 10 Genres')
fig.savefig(f'{PLOTS_DIR}/top_genres.png', bbox_inches='tight')
plt.close()

# 6. Top 10 Directors
fig, ax = plt.subplots()
top_dirs = df['Director'].value_counts().head(10)
sns.barplot(x=top_dirs.values, y=top_dirs.index, palette='magma', ax=ax)
ax.set_title('Top 10 Directors')
fig.savefig(f'{PLOTS_DIR}/top_directors.png', bbox_inches='tight')
plt.close()

# 7. Rating vs Votes (scatter)
fig, ax = plt.subplots()
ax.scatter(np.log1p(df['Votes']), df['Rating'], alpha=0.3, color='teal')
ax.set_title('Rating vs Votes (log)')
ax.set_xlabel('log(Votes)')
ax.set_ylabel('Rating')
fig.savefig(f'{PLOTS_DIR}/rating_vs_votes.png', bbox_inches='tight')
plt.close()

# 8. Rating vs Duration
fig, ax = plt.subplots()
ax.scatter(df['Duration'], df['Rating'], alpha=0.3, color='orange')
ax.set_title('Rating vs Duration')
ax.set_xlabel('Duration (min)')
ax.set_ylabel('Rating')
fig.savefig(f'{PLOTS_DIR}/rating_vs_duration.png', bbox_inches='tight')
plt.close()

# 9. Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr_cols = ['Rating', 'Votes', 'Duration', 'Year']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap')
fig.savefig(f'{PLOTS_DIR}/correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("All EDA plots saved to static/plots/")

# ============================================================
# PHASE 4 — FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print("PHASE 4: FEATURE ENGINEERING")
print("="*60)

# Movie Age
df['Movie_Age'] = CURRENT_YEAR - df['Year']

# Total Actors (non-unknown)
df['Total_Actors'] = (
    (df['Actor 1'] != 'Unknown').astype(int) +
    (df['Actor 2'] != 'Unknown').astype(int) +
    (df['Actor 3'] != 'Unknown').astype(int)
)

# Log Votes (reduces skewness)
df['Log_Votes'] = np.log1p(df['Votes'])

# Primary Genre (first genre only)
df['Primary_Genre'] = df['Genre'].str.split(',').str[0].str.strip()

print("New features created: Movie_Age, Total_Actors, Log_Votes, Primary_Genre")

# ============================================================
# PHASE 5 — ENCODING
# ============================================================
print("\n" + "="*60)
print("PHASE 5: LABEL ENCODING")
print("="*60)

# Label Encoding for high-cardinality categorical columns
le_dict = {}
cat_cols = ['Primary_Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Save label encoders
joblib.dump(le_dict, f'{MODELS_DIR}/label_encoders.pkl')
print(f"Label encoders saved for: {cat_cols}")

# ============================================================
# PHASE 6 — PREPARE FEATURES & SPLIT
# ============================================================
print("\n" + "="*60)
print("PHASE 6: PREPARING FEATURES")
print("="*60)

FEATURE_COLS = [
    'Duration', 'Log_Votes', 'Movie_Age', 'Total_Actors',
    'Primary_Genre_enc', 'Director_enc',
    'Actor 1_enc', 'Actor 2_enc', 'Actor 3_enc'
]

X = df[FEATURE_COLS].copy()
y = df['Rating']

# Fill any remaining NaNs in feature matrix
X.fillna(X.median(), inplace=True)

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"NaN in X: {X.isnull().sum().sum()}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
print("Scaler saved.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ============================================================
# PHASE 7 — MODEL TRAINING
# ============================================================
print("\n" + "="*60)
print("PHASE 7: MODEL TRAINING")
print("="*60)

models = {
    'Linear Regression'        : LinearRegression(),
    'Random Forest'            : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting'        : GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost'                  : XGBRegressor(n_estimators=100, random_state=42,
                                               verbosity=0, use_label_encoder=False)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    print(f"{name:30s} | MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")

# ============================================================
# PHASE 8 — EVALUATION TABLE
# ============================================================
print("\n" + "="*60)
print("PHASE 8: MODEL COMPARISON TABLE")
print("="*60)

results_df = pd.DataFrame(results).T.round(4)
print(results_df)

best_model_name = results_df['R2'].idxmax()
print(f"\n✅ Best Model: {best_model_name} (Highest R² Score)")

# ============================================================
# PHASE 9 — HYPERPARAMETER TUNING (Best Model)
# ============================================================
print("\n" + "="*60)
print("PHASE 9: HYPERPARAMETER TUNING")
print("="*60)

param_grid = {
    'n_estimators'  : [100, 200, 300],
    'max_depth'     : [4, 6, 8],
    'learning_rate' : [0.05, 0.1, 0.2],
    'subsample'     : [0.8, 1.0]
}

base_xgb = XGBRegressor(random_state=42, verbosity=0, use_label_encoder=False)

search = RandomizedSearchCV(
    base_xgb, param_grid, n_iter=15, cv=3,
    scoring='r2', random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
tuned_preds = best_model.predict(X_test)
tuned_r2    = r2_score(y_test, tuned_preds)
tuned_rmse  = np.sqrt(mean_squared_error(y_test, tuned_preds))

print(f"Best Params : {search.best_params_}")
print(f"Before Tuning → R2: {results['XGBoost']['R2']:.4f}  RMSE: {results['XGBoost']['RMSE']:.4f}")
print(f"After  Tuning → R2: {tuned_r2:.4f}  RMSE: {tuned_rmse:.4f}")

# ============================================================
# PHASE 10 — FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("PHASE 10: FEATURE IMPORTANCE")
print("="*60)

importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature'   : FEATURE_COLS,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feat_imp_df)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='Blues_r', ax=ax)
ax.set_title('Feature Importance (XGBoost Tuned)')
fig.savefig(f'{PLOTS_DIR}/feature_importance.png', bbox_inches='tight')
plt.close()
print("Feature importance plot saved.")

# ============================================================
# PHASE 11 — SAVE MODEL
# ============================================================
print("\n" + "="*60)
print("PHASE 11: SAVING MODEL")
print("="*60)

joblib.dump(best_model,  f'{MODELS_DIR}/model.pkl')
joblib.dump(FEATURE_COLS, f'{MODELS_DIR}/feature_cols.pkl')

print("✅ model.pkl saved")
print("✅ scaler.pkl saved")
print("✅ label_encoders.pkl saved")
print("✅ feature_cols.pkl saved")
print("\n🎉 TRAINING COMPLETE! Run app.py to start the web app.")