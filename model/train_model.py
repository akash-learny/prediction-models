import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
print("Loading training data...")
df = pd.read_csv('train_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# =========================
# DATA PREPARATION
# =========================
print("\n" + "="*50)
print("DATA PREPARATION")
print("="*50)

# Drop timestamp column and unnecessary features
columns_to_drop = ['_time']
if 'elapsed_minutes' in df.columns:
    columns_to_drop.append('elapsed_minutes')
if 'Weighted_Temp' in df.columns:
    columns_to_drop.append('Weighted_Temp')

df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Display basic statistics
print(f"\nTarget variable (Doneness_Score) statistics:")
print(df['Doneness_Score'].describe())

# =========================
# FEATURE ENGINEERING
# =========================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Create additional features (without Weighted_Temp dependency)
df['Temp_Lid_Ambient_Diff'] = df['T1--Temp_01_Lid'] - df['T1--Temp_05_Ambient_Temp']
df['Temp_Handle_Ambient_Diff'] = df['T1--Temp_03_Handle'] - df['T1--Temp_05_Ambient_Temp']
df['Temp_InnerWall_Ambient_Diff'] = df['T1--Temp_04_InnerWall'] - df['T1--Temp_05_Ambient_Temp']
df['Temp_Range'] = df[['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 'T1--Temp_04_InnerWall']].max(axis=1) - df[['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 'T1--Temp_04_InnerWall']].min(axis=1)

print(f"New features created: {['Temp_Lid_Ambient_Diff', 'Temp_Handle_Ambient_Diff', 'Temp_InnerWall_Ambient_Diff', 'Temp_Range']}")
print(f"Total features: {df.shape[1] - 1}")  # -1 for target variable

# =========================
# PREPARE X AND Y
# =========================
# Separate features and target
X = df.drop(columns=['Doneness_Score'])
y = df['Doneness_Score']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nFeatures used:\n{X.columns.tolist()}")

# =========================
# TRAIN-TEST SPLIT
# =========================
print("\n" + "="*50)
print("TRAIN-TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# =========================
# LINEAR REGRESSION MODEL
# =========================
print("\n" + "="*50)
print("LINEAR REGRESSION MODEL")
print("="*50)

model = LinearRegression()
print("Training Linear Regression model...")
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

print(f"\nTraining R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"\n5-Fold Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =========================
# MODEL PERFORMANCE SUMMARY
# =========================
print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)

summary = pd.DataFrame({
    'Metric': ['Train R²', 'Test R²', 'MAE', 'RMSE', 'MAPE (%)', 'CV Mean R²'],
    'Value': [train_r2, test_r2, mae, rmse, mape, cv_scores.mean()]
})

print("\n", summary.to_string(index=False))

# =========================
# SAVE MODEL
# =========================
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

joblib.dump(model, 'rice_doneness_model.pkl')

# Save feature names for later use
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

# Save model metadata
model_metadata = {
    'model_type': 'Linear Regression',
    'train_r2': train_r2,
    'test_r2': test_r2,
    'mae': mae,
    'rmse': rmse,
    'mape': mape,
    'cv_mean_r2': cv_scores.mean(),
    'cv_std_r2': cv_scores.std(),
    'n_features': X.shape[1],
    'n_samples': X.shape[0]
}
joblib.dump(model_metadata, 'model_metadata.pkl')

print("✅ Model saved successfully:")
print("   - rice_doneness_model.pkl")
print("   - feature_names.pkl")
print("   - model_metadata.pkl")

# =========================
# VISUALIZATIONS
# =========================
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=10, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Doneness Score')
axes[0, 0].set_ylabel('Predicted Doneness Score')
axes[0, 0].set_title(f'Actual vs Predicted\nR² = {test_r2:.4f}, MAE = {mae:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Plot
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, s=10, color='blue')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Doneness Score')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals Distribution
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='blue')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].grid(True, alpha=0.3)

# 4. Prediction Error
axes[1, 1].scatter(range(len(y_test)), y_test.values, alpha=0.6, s=20, label='Actual', color='green')
axes[1, 1].scatter(range(len(y_test)), y_test_pred, alpha=0.6, s=20, label='Predicted', color='blue')
axes[1, 1].set_xlabel('Sample Index')
axes[1, 1].set_ylabel('Doneness Score')
axes[1, 1].set_title('Actual vs Predicted Values')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: model_evaluation.png")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
