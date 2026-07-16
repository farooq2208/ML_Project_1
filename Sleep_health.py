"""Predicting Sleep Health Based on Life-Style"""

# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Loading Dataset
df = pd.read_csv('sleep_health.csv')

# Detecting Missing Values
df.isna().sum()     # After Printing
# "sleep_disorder" contains 219 Missing Values,
# these are not actually Missing Values, but "None" due to No-disorder is sleep.
# So, we will fill these with "Healthy"

# Imputing Missing Values with "Healthy"
df.fillna('Healthy', inplace = True)

# Dropping Irrelavant Column
df.drop(columns=['Person_ID'], inplace = True)

# We have used map() to map categories in target to 0, 1 and 2.
df['sleep_disorder'] =  df['sleep_disorder'].map({'Sleep Apnea':0,'Insomnia':1 ,'Healthy':2})

# Separating Data into features (X) and target (y)
X = df.drop(columns=['sleep_disorder'])
y = df['sleep_disorder']

# Splitting features and target in train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# Testing whether classes are balanced or imbalanced
y_train.value_counts(normalize=True) * 100      # Reult is Imbalanced

# Selecting Numerical and Categorical (Nominal, Ordinal) Columns
num = X.select_dtypes(include = 'number').columns
nominal_cat = X.select_dtypes(exclude = 'number').drop(columns=['BMI_category']).columns
ordinal_cat = ['BMI_category']

# Creating Numerical and Categorical(Nominal, Ordinal) Pipelines
num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
nominal_cat_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore') )])
ordinal_cat_pipeline = Pipeline(steps=[('encoder', OrdinalEncoder(categories=[['Normal', 'Overweight','Obese']]))])

# Column Transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num),
    ('nominal_cat', nominal_cat_pipeline, nominal_cat),
    ('ordinal_cat', ordinal_cat_pipeline, ordinal_cat)
])

# Final_pipeline
pipe = Pipeline(steps=[('preprocessor', preprocessor),('model', XGBClassifier())])

# Parameters Grid to Try
param_grid = [
    {'model': [LogisticRegression(max_iter=1000)],
     'model__C': [0.1, 1, 10, 100],
     'model__penalty': ['l2'],
     'model__solver': ['lbfgs']},

    {'model': [DecisionTreeClassifier()],
     'model__max_depth': [5, 10, 20, None],
     'model__min_samples_split': [2, 5, 10]},

    {'model': [RandomForestClassifier()],
     'model__n_estimators': [100, 300],
     'model__max_depth': [5, 10, None]},

    {'model': [XGBClassifier(eval_metric='mlogloss')],
     'model__n_estimators': [100, 300],
     'model__max_depth': [3, 6, 10],
     'model__learning_rate': [0.01, 0.1, 0.3]}
]

# Determining Best Estimator
grid = GridSearchCV(pipe, param_grid, cv=5 , n_jobs=-1, scoring='f1_weighted' )     # f1_weighted , used because classes are imbalanced
model = grid.fit(X_train, y_train).best_estimator_

# Testing Best Estimator on Training and Test Data
y_pred = model.predict(X_test)

# Classifying whether Model is underfitting/overfitting
print(f'Training Score: {accuracy_score(y_train, model.predict(X_train))}')
print(f'Testing Score: {accuracy_score(y_test, model.predict(X_test))}')

# Evaluation Metrics
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print(f'Precision Score: {precision_score(y_test, y_pred, average="weighted")}')
print(f'Recall Score: {recall_score(y_test, y_pred, average='weighted')}')
print(f'F1-Score: {f1_score(y_test, y_pred, average = 'weighted')}')

# Saving Model
joblib.dump(model, 'sleep_disorder_model.pkl')