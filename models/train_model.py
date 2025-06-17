# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ..src.model_utils import evaluate_model, show_plots, save_model

# Load & prepare data
df = pd.read_csv("data/processed/final_training_data.csv", dtype={19: str, 20: str}, low_memory=False)

# Drop irrelevant cols
df = df.drop(columns=[
    'requestid', 'wktgeom', 'street', 'fromst', 'tost',
    'holiday_name', 'holiday_type', 'timestamp', 'weather_description'
])

y = df['vol']
X = df.drop(columns='vol')

categorical_features = ['boro', 'direction', 'dayofweek']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate and plot
mae, rmse, r2 = evaluate_model(y_test, y_pred)
print(f"[Hold-out Test Set] MAE: {mae:.2f}")
print(f"[Hold-out Test Set] RMSE: {rmse:.2f}")
print(f"[Hold-out Test Set] R²: {r2:.4f}")
show_plots(y_test, y_pred)

# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
rmse_scores = -cv_scores

print("\n[TimeSeriesSplit CV Evaluation]")
print("Fold RMSE scores:", rmse_scores)
print("Mean RMSE: {:.2f}".format(rmse_scores.mean()))
print("Standard Deviation: {:.2f}".format(rmse_scores.std()))

# Save model
save_model(pipeline)