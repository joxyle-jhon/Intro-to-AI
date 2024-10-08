import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

df_d.head()

df_d.info()

df_d.describe()

print(df_d.columns)

X = df_d.drop(['target'], axis=1)
y = df_d.target

from sklearn.model_selection import train_test_split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, random_state=123, test_size=0.2)

#Losgistic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

lrc = LogisticRegression().fit(X_train2, y_train2)
lrc_preds = lrc.predict(X_test2)
print(f'Logistic Regression Accuracy: {accuracy_score(y_test2.values, lrc_preds) * 100}%')

conf_matrix = confusion_matrix(y_test2.values, lrc_preds)
classification_rep = classification_report(y_test2.values, lrc_preds)

print(f"Confusion Matrix: \n{conf_matrix}")
print(f"Classification Report:\n {classification_rep}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc = DecisionTreeClassifier(random_state=123).fit(X_train2,y_train2)
dtc_preds = dtc.predict(X_test2)

print(f'Decision Tress Classifier Accurracy: {accuracy_score(y_test2.values, dtc_preds) * 100}')

conf_matrix = confusion_matrix(y_test2.values, dtc_preds)
classification_rep = classification_report(y_test2.values, dtc_preds)

print (f'Confusion Matric: \n {conf_matrix}')
print(f'Classification Report: \n {classification_rep}')

# The analysis of the Breast Cancer Wisconsin dataset reveals high performance in binary classification tasks.
# Logistic Regression achieved an accuracy of 98.25%,
# outperforming the Decision Tree Classifier, which had an accuracy of 95.61%.
# This indicates that Logistic Regression is more effective for this dataset,
# emphasizing the significance of model selection in achieving optimal results in breast cancer prediction.

