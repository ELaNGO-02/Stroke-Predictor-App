# Stroke Prediction ML Project
# Tasks: 1) Dataset selection & problem definition
#        2) Data cleaning & EDA
#        3) Model building (compare 2 models)
#        4) Deployment (Streamlit app skeleton)

"""
Instructions:
1. Download the Kaggle Stroke Prediction dataset CSV from:
   https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
   Common filename: 'healthcare-dataset-stroke-data.csv'
2. Put the CSV in the same folder as this script/notebook.
3. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib streamlit
4. Run this script in order. It will save plots as PNGs and the best model as 'best_stroke_model.joblib'.
5. To run the Streamlit app produced here: `streamlit run app.py`

"""

# ---------- TASK 1: Problem Definition ----------
# Problem: Predict whether a patient will have a stroke (binary classification)
# Inputs: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
#         avg_glucose_level, bmi, smoking_status
# Output: stroke (0/1)

# ---------- Imports ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

# ---------- Load data ----------
DATA_FILE = 'healthcare-dataset-stroke-data.csv'
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Please download the dataset CSV from Kaggle and place it as '{DATA_FILE}' in this folder.")

df = pd.read_csv(DATA_FILE)
print('Original shape:', df.shape)
print(df.head())

# ---------- TASK 2: Data Cleaning & EDA ----------
# 1) Basic info
print('\nInfo:')
print(df.info())
print('\nDescribe:')
print(df.describe(include='all'))

# 2) Drop irrelevant columns (id)
if 'id' in df.columns:
    df = df.drop(columns=['id'])
    print('\nDropped id column. New shape:', df.shape)

# 3) Missing values
print('\nMissing values per column:')
print(df.isnull().sum())

# BMI has missing values (if any) - inspect
if 'bmi' in df.columns:
    print('\nBMI missing count:', df['bmi'].isnull().sum())

# 4) Duplicates
dups = df.duplicated().sum()
print(f'Number of duplicate rows: {dups}')
if dups>0:
    df = df.drop_duplicates()
    print('Dropped duplicates. New shape:', df.shape)

# 5) Outliers - look at numeric distributions
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('\nNumeric columns:', numeric_cols)

# Create at least 5 plots and save them
os.makedirs('plots', exist_ok=True)

# Histogram grid
plt.figure(figsize=(12,8))
for i, col in enumerate(['age','avg_glucose_level','bmi']):
    if col in df.columns:
        plt.subplot(2,2,i+1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(col)
plt.tight_layout()
plt.savefig('plots/histograms_1.png')
plt.clf()

# Count plots for categorical
plt.figure(figsize=(12,8))
cat_cols = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']
for i, col in enumerate(cat_cols[:4]):
    if col in df.columns:
        plt.subplot(2,2,i+1)
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(col)
plt.tight_layout()
plt.savefig('plots/counts_1.png')
plt.clf()

plt.figure(figsize=(12,8))
for i, col in enumerate(cat_cols[4:]):
    if col in df.columns:
        plt.subplot(2,2,i+1)
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(col)
plt.tight_layout()
plt.savefig('plots/counts_2.png')
plt.clf()

# Correlation heatmap (for numeric)
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Numeric feature correlation')
plt.tight_layout()
plt.savefig('plots/corr_heatmap.png')
plt.clf()

# Boxplots for numeric columns to see outliers
plt.figure(figsize=(12,6))
for i, col in enumerate(['age','avg_glucose_level','bmi']):
    if col in df.columns:
        plt.subplot(1,3,i+1)
        sns.boxplot(x=df[col].dropna())
        plt.title('Boxplot ' + col)
plt.tight_layout()
plt.savefig('plots/boxplots.png')
plt.clf()

# Target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='stroke', data=df)
plt.title('Stroke distribution (0 = no, 1 = yes)')
plt.tight_layout()
plt.savefig('plots/target_count.png')
plt.clf()

# Pairplot sample (small subset to keep it fast)
if set(['age','avg_glucose_level','bmi','stroke']).issubset(df.columns):
    sample = df[['age','avg_glucose_level','bmi','stroke']].dropna().sample(n=min(300, len(df)), random_state=42)
    sns.pairplot(sample, hue='stroke')
    plt.savefig('plots/pairplot_sample.png')
    plt.clf()

print('Saved EDA plots in ./plots/')

# 6) Handle missing values
# For bmi we will impute median; other columns should be complete
if df['bmi'].isnull().sum()>0:
    median_bmi = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(median_bmi)
    print(f'Imputed bmi missing values with median: {median_bmi}')

# 7) Outlier treatment (winsorize numerics at 1st and 99th percentiles)
for col in ['age','avg_glucose_level','bmi']:
    if col in df.columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)
print('Applied winsorization to numeric features.')

# 8) Encoding categorical variables
# Identify categorical features
cat_features = [c for c in df.columns if df[c].dtype == 'object']
num_features = [c for c in df.columns if c not in cat_features + ['stroke']]
print('\nCategorical features:', cat_features)
print('Numeric features:', num_features)

# One-hot encode categorical features, scale numerics in pipeline later

# ---------- TASK 3: Model Building ----------
# Prepare X, y
X = df.drop('stroke', axis=1)
y = df['stroke']

# Train/test split (stratify because target is imbalanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('\nTrain shape:', X_train.shape, 'Test shape:', X_test.shape)

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# Model pipelines
pipe_lr = Pipeline(steps=[('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
pipe_rf = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])

# Fit Logistic Regression
print('\nTraining Logistic Regression...')
pipe_lr.fit(X_train, y_train)
pred_lr = pipe_lr.predict(X_test)

# Fit Random Forest
print('\nTraining Random Forest...')
pipe_rf.fit(X_train, y_train)
pred_rf = pipe_rf.predict(X_test)

# Metrics function
from sklearn.metrics import confusion_matrix

def print_metrics(y_true, y_pred, label='Model'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{label} metrics:\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}")
    print('\nClassification report:\n', classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion matrix - ' + label)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_{label}.png')
    plt.clf()

print_metrics(y_test, pred_lr, 'LogisticRegression')
print_metrics(y_test, pred_rf, 'RandomForest')

# Compare by F1
f1_lr = f1_score(y_test, pred_lr, zero_division=0)
f1_rf = f1_score(y_test, pred_rf, zero_division=0)

if f1_rf >= f1_lr:
    best_model = pipe_rf
    best_name = 'RandomForest'
else:
    best_model = pipe_lr
    best_name = 'LogisticRegression'

print(f"\nBest model selected: {best_name}")

# Save the best model
joblib.dump(best_model, 'best_stroke_model.joblib')
print('Saved best model to best_stroke_model.joblib')

# ---------- TASK 4: Deployment (Streamlit app) ----------
# Write a simple Streamlit app (app.py). Run: streamlit run app.py
streamlit_app = r"""
import streamlit as st
import pandas as pd
import joblib

st.title('Stroke Prediction Demo')
model = joblib.load('best_stroke_model.joblib')

st.sidebar.header('Patient input')
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=45)
gender = st.sidebar.selectbox('Gender', ['Male','Female','Other'])
hypertension = st.sidebar.selectbox('Hypertension', [0,1])
heart_disease = st.sidebar.selectbox('Heart disease', [0,1])
ever_married = st.sidebar.selectbox('Ever married', ['Yes','No'])
work_type = st.sidebar.selectbox('Work type', ['children','Govt_job','Never_worked','Private','Self-employed'])
Residence_type = st.sidebar.selectbox('Residence type', ['Urban','Rural'])
avg_glucose_level = st.sidebar.number_input('Average glucose level', min_value=40.0, max_value=400.0, value=100.0)
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0)
smoking_status = st.sidebar.selectbox('Smoking status', ['never smoked','formerly smoked','smokes','Unknown'])

input_dict = {
    'gender':[gender], 'age':[age], 'hypertension':[hypertension], 'heart_disease':[heart_disease],
    'ever_married':[ever_married], 'work_type':[work_type], 'Residence_type':[Residence_type],
    'avg_glucose_level':[avg_glucose_level], 'bmi':[bmi], 'smoking_status':[smoking_status]
}
input_df = pd.DataFrame(input_dict)

if st.button('Predict'):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0,1] if hasattr(model, 'predict_proba') else None
    if pred==1:
        st.error(f'Prediction: Stroke likely (probability {proba:.2f})')
    else:
        st.success(f'Prediction: No stroke predicted (probability {proba:.2f})')
"""

with open('app.py','w') as f:
    f.write(streamlit_app)

print('Wrote Streamlit app to app.py')
print('\nAll artifacts:')
print(' - EDA plots: ./plots/')
print(' - Saved model: best_stroke_model.joblib')
print(' - Streamlit app: app.py')

# End of script
print('\nCompleted all tasks for Stroke Prediction project.')

