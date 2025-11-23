import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.xgboost  # Use this if the best model is XGBoost; change to mlflow.sklearn if it's another model
import numpy as np

# Load data
df = pd.read_csv('C:\\Users\\ruchi\\OneDrive\\Desktop\\streamlit_app\\env\\Scripts\\cleaned_medical_insurance.csv')  # Adjust path if needed

# Load the registered model from MLflow Model Registry
# Assuming the best model is XGBoost (from training script); change to mlflow.sklearn if it's Linear/Random Forest
try:
    model = mlflow.xgboost.load_model("models:/Best_Insurance_Model/Production")  # Correct URI for registered model
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}. Ensure the model is registered in MLflow.")
    st.stop()  # Stop the app if model can't load

st.title("Medical Insurance Cost Prediction")

# EDA Section
st.header("EDA Insights")
st.subheader("Charges Distribution")
fig, ax = plt.subplots()
sns.histplot(df['charges'], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Charges by Smoking Status")
fig, ax = plt.subplots()
sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
st.pyplot(fig)

# Prediction Section
st.header("Predict Your Insurance Cost")
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Gender", ["female", "male"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode inputs (must match preprocessing in data_preprocessing.py)
sex_enc = 0 if sex == "female" else 1
smoker_enc = 0 if smoker == "no" else 1
# One-hot for region (drop 'northeast' as first, so northwest, southeast, southwest)
region_enc = [
    1 if region == "northwest" else 0,
    1 if region == "southeast" else 0,
    1 if region == "southwest" else 0
]
# BMI category (0: underweight, 1: normal, 2: overweight, 3: obese)
bmi_cat = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
# Interaction terms
age_smoker = age * smoker_enc
bmi_smoker = bmi * smoker_enc

# Prepare input as numpy array (match feature order from training)
input_data = np.array([[age, sex_enc, bmi, children, smoker_enc, *region_enc, bmi_cat, age_smoker, bmi_smoker]])
prediction = model.predict(input_data)[0]
st.write(f"Estimated Cost: ${prediction:.2f}")
st.write("Note: This is an estimate; actual costs vary. (Optional: Add confidence intervals if your model supports it.)")