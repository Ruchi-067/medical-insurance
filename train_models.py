import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

# Load processed data
df = pd.read_csv('C:\\Users\\ruchi\\OneDrive\\Desktop\\streamlit_app\\env\\Scripts\\cleaned_medical_insurance.csv')
preprocessor = joblib.load('C:\\Users\\ruchi\\OneDrive\\Desktop\\streamlit_app\\env\\preprocessor.pkl')

X = df.drop('charges', axis=1)
y = df['charges']
X_processed = preprocessor.transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment('Medical Insurance Prediction')

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

best_model = None
best_rmse = float('inf')

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_param('model', name)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('R2', r2)
        mlflow.sklearn.log_model(model, 'model')
        
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

# Register best model (manually in MLflow UI or via API)
print(f"Best model: {best_model.__class__.__name__} with RMSE={best_rmse:.2f}")
print("Training complete. Check MLflow UI for logs.")