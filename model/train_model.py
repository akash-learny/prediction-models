import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import matplotlib.pyplot as plt
import os
import toml

# =========================
# LOAD PATH FROM SECRETS
# =========================
SECRETS_PATH = os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml')
secrets = toml.load(SECRETS_PATH)
DATA_PATH = secrets['paths']['TRAIN_DATASET']

# Resolve relative to workspace root (one level up from model/)
if not os.path.isabs(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', DATA_PATH)

# =========================
# LOAD DATA
# =========================
print("Loading training data...")
print(f"Path: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# =========================
# DATA PREPARATION
# =========================
print("\n" + "="*50)
print("DATA PREPARATION")
print("="*50)

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget variable (doneness) statistics:")
print(df['doneness'].describe())

# =========================
# FEATURE ENGINEERING
# =========================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Derived features (matching program.ipynb)
df['thermal_energy']    = df['power'] * df['cook_time']
df['water_absorption']  = df['water_ratio'] * df['soak_time']
df['temp_effect']       = (df['initial_temp'] - 20) / 15
df['power_time']        = df['power'] * df['cook_time']

print(f"Engineered features: thermal_energy, water_absorption, temp_effect, power_time")

# One-hot encode rice_type
df = pd.get_dummies(df, columns=['rice_type'])
print(f"One-hot encoded rice_type columns: {[c for c in df.columns if c.startswith('rice_type')]}")

# =========================
# PREPARE X AND Y
# =========================
X = df.drop(columns=['doneness'])
y = df['doneness']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# =========================
# TRAIN-TEST SPLIT
# =========================
print("\n" + "="*50)
print("TRAIN-TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# =========================
# LINEAR REGRESSION (baseline)
# =========================
print("\n" + "="*50)
print("LINEAR REGRESSION (baseline)")
print("="*50)

lin = LinearRegression()
lin.fit(X_train, y_train)
pred_lin = lin.predict(X_test)
rmse_lin = np.sqrt(mean_squared_error(y_test, pred_lin))
r2_lin   = r2_score(y_test, pred_lin)
print(f"Linear RMSE: {rmse_lin:.4f} | R²: {r2_lin:.4f}")

# =========================
# RANDOM FOREST (primary model)
# =========================
print("\n" + "="*50)
print("RANDOM FOREST REGRESSOR")
print("="*50)

rf = RandomForestRegressor(n_estimators=150, random_state=42)
print("Training Random Forest model...")
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_test_pred  = rf.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2  = r2_score(y_test,  y_test_pred)
mae      = mean_absolute_error(y_test, y_test_pred)
rmse     = np.sqrt(mean_squared_error(y_test, y_test_pred))
# sMAPE: symmetric MAPE — stable when actual values are near zero
mape = np.mean(np.abs(y_test.values - y_test_pred) / (np.abs(y_test.values) + np.abs(y_test_pred) + 1e-6) * 2) * 100

print(f"\nTraining R²:  {train_r2:.4f}")
print(f"Test R²:      {test_r2:.4f}")
print(f"MAE:          {mae:.4f}")
print(f"RMSE:         {rmse:.4f}")
print(f"sMAPE:        {mape:.2f}%")

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f"\n5-Fold CV R²: {cv_scores}")
print(f"Mean CV R²:   {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# =========================
# SAVE MODEL ARTIFACTS
# =========================
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

OUT_DIR = os.path.dirname(__file__)

joblib.dump(rf,                    os.path.join(OUT_DIR, 'rice_doneness_model.pkl'))
joblib.dump(X.columns.tolist(),    os.path.join(OUT_DIR, 'feature_names.pkl'))

model_metadata = {
    'model_type':   'Random Forest Regressor',
    'n_estimators': 150,
    'train_r2':     train_r2,
    'test_r2':      test_r2,
    'mae':          mae,
    'rmse':         rmse,
    'mape':         mape,   # sMAPE (%)
    'cv_mean_r2':   cv_scores.mean(),
    'cv_std_r2':    cv_scores.std(),
    'n_features':   X.shape[1],
    'n_samples':    X.shape[0],
    'feature_names': X.columns.tolist(),
    'target':       'doneness (0-100)',
}
joblib.dump(model_metadata, os.path.join(OUT_DIR, 'model_metadata.pkl'))

print("Model saved:  rice_doneness_model.pkl")
print("Features:     feature_names.pkl")
print("Metadata:     model_metadata.pkl")

# =========================
# VISUALIZATIONS
# =========================
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=10, color='steelblue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Doneness Score')
axes[0, 0].set_ylabel('Predicted Doneness Score')
axes[0, 0].set_title(f'Actual vs Predicted\nR² = {test_r2:.4f}, MAE = {mae:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Plot
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, s=10, color='steelblue')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Doneness Score')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals Distribution
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution')
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1, 1], color='steelblue')
axes[1, 1].set_title('Feature Importance')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
print("Visualization saved: model_evaluation.png")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
