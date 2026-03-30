import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,f1_score
from sklearn.model_selection import cross_val_score
import pickle

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Display basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Handle zeros in the dataset - in this dataset, zeros in certain columns (like Glucose, BloodPressure, etc.) are considered invalid and will be treated as missing values
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)

for col in cols_with_zeros:
    data[col] = data[col].fillna(data[col].median())
print((data[cols_with_zeros]==0).sum())

# Handle outliers (if necessary) - for simplicity, we will use the IQR method to identify and remove outliers from the numerical columns
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

data['Is_Obese'] = (data['BMI'] >= 30).astype(int)

# Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Define the preprocessing steps for numerical and categorical features
num_transformer=Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ]
)
cat_transformer=Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder())
    ]
)
preprocessor=ColumnTransformer(
    transformers=[
        ('num',num_transformer,X.select_dtypes(include=np.number).columns),
        ('cat',cat_transformer,X.select_dtypes(exclude=np.number).columns)
    ]
)
# Create a pipeline that combines the preprocessor with a Random Forest Classifier
pipeline=Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier',RandomForestClassifier(
    max_depth=23,
    min_samples_split=12,
    n_estimators=187,
    n_jobs=-1,
    verbose=2,
    random_state=42
))
    ]
)
#test the pipeline on the entire dataset before splitting
print("Shape of data:", data.shape)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
# Split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
# Make predictions on the test set
y_pred = pipeline.predict(X_test)
# Evaluate the model

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print('Classification Report:')
print(report)
# Save the trained model to a file
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved to diabetes_model.pkl")

