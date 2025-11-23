import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv('C:\\Users\\ruchi\\OneDrive\\Desktop\\streamlit_app\\medical_insurance.csv')

# Clean data: Check and remove duplicates (no major missing values in this dataset)
print(f"Original shape: {df.shape}")
print(f"Duplicates: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"After cleaning: {df.shape}")

# Feature engineering
def bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif 18.5 <= bmi < 25:
        return 'normal'
    elif 25 <= bmi < 30:
        return 'overweight'
    else:
        return 'obese'

df['bmi_category'] = df['bmi'].apply(bmi_category)
df['age_smoker'] = df['age'] * df['smoker'].map({'yes': 1, 'no': 0})  # Interaction term

# Define preprocessor
categorical_features = ['sex', 'smoker', 'region', 'bmi_category']
numerical_features = ['age', 'bmi', 'children', 'age_smoker']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Prepare X and y
X = df.drop('charges', axis=1)
y = df['charges']
X_processed = preprocessor.fit_transform(X)

# Save cleaned dataset and preprocessor (for app use)
df.to_csv('cleaned_medical_insurance.csv', index=False)
import joblib
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessing complete. Files saved: cleaned_medical_insurance.csv, preprocessor.pkl")