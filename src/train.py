import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os
import sys
from CustomTransformers import FeatureAdder

sys.path

cwd = os.getcwd()


dataset_path = os.path.join(cwd,'input','card_transdata.csv')
df = pd.read_csv(dataset_path)
df.head()
    
fe = FeatureAdder(add_columns=['distance_from_home', 'distance_from_last_transaction'],new_col = 'distance_sum')
df = fe.transform(df)

numerical_columns = ['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price', 'distance_sum']
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

X = df.drop(columns = ['fraud'])
y = df['fraud']

X_train,X_test,y_train,y_test = train_test_split(X , y,  test_size = 0.2, random_state = 42)
smote = SMOTE(random_state = 42)

X_train_res , y_train_res = smote.fit_resample(X_train, y_train)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

df.head()

# param_grid = {
#     'n_estimators': [20, 30],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.7, 0.8, 0.9],
#     'colsample_bytree': [0.7, 0.8, 0.9]
# }

# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X2_train_res, y2_train_res)

# best_params = grid_search.best_params_
# print(f"Best parameters: {best_params}")

# best_xgb = grid_search.best_estimator_
xgb.fit(X_train_res, y_train_res)

y_pred = xgb.predict(X_test)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report = classification_report(y_test, y_pred)

print("Classification Report:\n", report)

f1_score_macro = report_dict['macro avg']['f1-score']
print(f"Macro F1-score: {f1_score_macro}")

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(fe,scaler, xgb)
pipeline.fit(X_train,y_train)

import pickle

models_folder = os.path.join(cwd, 'models')

pipeline_path = os.path.join(models_folder, 'new_pipeline.pkl')

with open(pipeline_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"Pipeline saved to {pipeline_path}")