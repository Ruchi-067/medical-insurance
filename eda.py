import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('cleaned_medical_insurance.csv')

# Set style
sns.set(style='whitegrid')

# 1. Univariate Analysis
print("Univariate Analysis:")
print(f"Average BMI: {df['bmi'].mean():.2f}")
plt.figure(figsize=(8, 5))
sns.histplot(df['charges'], kde=True)
plt.title('Distribution of Medical Insurance Charges')
plt.savefig('charges_distribution.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['age'])
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='smoker', data=df)
plt.title('Smokers vs Non-Smokers')
plt.savefig('smoker_count.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='region', data=df)
plt.title('Policyholders by Region')
plt.savefig('region_count.png')
plt.show()

# 2. Bivariate Analysis
print("Bivariate Analysis:")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='age', y='charges', data=df)
plt.title('Charges vs Age')
plt.savefig('charges_vs_age.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Charges by Smoking Status')
plt.savefig('charges_by_smoker.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='bmi', y='charges', data=df)
plt.title('Charges vs BMI')
plt.savefig('charges_vs_bmi.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='sex', y='charges', data=df)
plt.title('Charges by Gender')
plt.savefig('charges_by_gender.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='children', y='charges', data=df)
plt.title('Charges vs Number of Children')
plt.savefig('charges_vs_children.png')
plt.show()

# 3. Multivariate Analysis
print("Multivariate Analysis:")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='age', y='charges', hue='smoker', data=df)
plt.title('Charges vs Age by Smoking Status')
plt.savefig('charges_age_smoker.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='region', y='charges', hue='sex', data=df[df['smoker'] == 'yes'])
plt.title('Charges by Region and Gender (Smokers Only)')
plt.savefig('charges_region_gender_smoker.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.pairplot(df[['age', 'bmi', 'charges']], hue='smoker')
plt.title('Age, BMI, Charges by Smoking Status')
plt.savefig('pairplot_age_bmi_charges.png')
plt.show()

obese_smokers = df[(df['bmi'] > 30) & (df['smoker'] == 'yes')]
print(f"Average charges for obese smokers: ${obese_smokers['charges'].mean():.2f}")

# 4. Outlier Detection
print("Outlier Detection:")
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['charges'])
plt.title('Outliers in Charges')
plt.savefig('outliers_charges.png')
plt.show()

extreme_bmi = df[df['bmi'] > 50]
print(f"Extreme BMI individuals: {len(extreme_bmi)}")

# 5. Correlation Analysis
print("Correlation Analysis:")
numeric_df = df[['age', 'bmi', 'children', 'charges']]
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

print("EDA complete. Visualizations saved as PNG files.")