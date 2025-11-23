import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import joblib

# Load data and model
df = pd.read_csv('C:\\Users\\ruchi\\OneDrive\\Desktop\\streamlit_app\\env\\Scripts\\cleaned_medical_insurance.csv')
model = mlflow.sklearn.load_model('models:/Best Model/Production')  # Update with your registered model name/version
preprocessor = joblib.load('C:\\Users\\ruchi\\OneDrive\\Desktop\\streamlit_app\\env\\preprocessor.pkl')

# BMI category function
def bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif 18.5 <= bmi < 25:
        return 'normal'
    elif 25 <= bmi < 30:
        return 'overweight'
    else:
        return 'obese'

st.title('Medical Insurance Cost Prediction')

# EDA Section
st.header('EDA Insights')
eda_options = st.multiselect('Select EDA Visualizations', 
                             ['Charges Distribution', 'Charges vs Age by Smoker', 'Correlation Matrix'])

if 'Charges Distribution' in eda_options:
    fig, ax = plt.subplots()
    sns.histplot(df['charges'], kde=True, ax=ax)
    st.pyplot(fig)

if 'Charges vs Age by Smoker' in eda_options:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker', ax=ax)
    st.pyplot(fig)

if 'Correlation Matrix' in eda_options:
    numeric_df = df[['age', 'bmi', 'children', 'charges']]
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Prediction Section
st.header('Predict Your Insurance Cost')
age = st.slider('Age', 18, 100, 30)
sex = st.selectbox('Gender', ['male', 'female'])
bmi = st.slider('BMI', 10.0, 50.0, 25.0)
children = st.slider('Number of Children', 0, 10, 0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

if st.button('Predict'):
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
        'smoker': [smoker], 'region': [region], 
        'bmi_category': [bmi_category(bmi)], 
        'age_smoker': [age * (1 if smoker == 'yes' else 0)]
    })
    
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]
    st.write(f'Estimated Insurance Cost: ${prediction:.2f}')
    
    # Optional error margin (based on MAE from training)
    mae = 4000  # Replace with actual MAE from logs
    st.write(f'Approximate Range: ${max(0, prediction - mae):.2f} - ${prediction + mae:.2f}')