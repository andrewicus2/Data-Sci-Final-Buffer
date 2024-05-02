
import pandas as pd
import seaborn as sn
# Commented out IPython magic to ensure Python compatibility.
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# %matplotlib inline

df = pd.read_csv("ifood-data.csv")

df.head()

df['Education'] = df['Education'].astype('category').cat.codes
df['Marital_Status'] = df['Marital_Status'].astype('category').cat.codes

df = df.drop(["Dt_Customer"], axis = 1)
df = df.drop(["ID"], axis = 1)

df = df.dropna()

df.head()

X = df.drop(labels = ['Response'], axis = 1)
y = df["Response"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

prediction = logmodel.predict(X_test)

print(classification_report(y_test,prediction))

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_cols = X.columns
feature_cols

from sklearn.tree import export_graphviz
feature_names = X.columns
dot_data = export_graphviz(clf, out_file=None,

                         feature_names=feature_cols,

                         class_names=['0','1'],

                         filled=True, rounded=True,

                         special_characters=True)

graph = graphviz.Source(dot_data)
graph

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import graphviz
from sklearn.tree import export_graphviz
feature_names = X.columns
dot_data = export_graphviz(clf, out_file=None,

                         feature_names=feature_cols,

                         class_names=['0','1'],

                         filled=True, rounded=True,

                         special_characters=True)

graph = graphviz.Source(dot_data)
graph

from shapash.explainer.smart_explainer import SmartExplainer

xpl = SmartExplainer(clf)

y_pred = pd.Series(y_pred)
X_test = X_test.reset_index(drop=True)
xpl.compile(x=X_test, y_pred=y_pred)

xpl.plot.features_importance()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

results = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, results))

# Import necessary libraries
import numpy as np # a Python library used for working with arrays
import pandas as pd # it allows us to analyze big data and make conclusions based on statistical theories

from pycaret.datasets import get_data #  allows you to easily access and load built-in datasets for machine learning experimentation
from pycaret.classification import * #  imports all the classification-related functions
from sklearn.model_selection import train_test_split # This function is commonly used to split a dataset into training and testing subsets.
import mlflow # MLflow is an open-source platform for managing the machine learning lifecycle
from sklearn import metrics as sk_metrics # imports the metrics module from the sklearn library and use various evaluation metrics and scoring functions provided by sci)kit

# Split data into training and testing sets
loan_train, loan_test = train_test_split(df, test_size=0.2, random_state=42)

# Initialize PyCaret setup with the training set
cls1 = setup(data = loan_train, target = 'Response')

# Compare all models and select top 3
top3 = compare_models(include=['lr', 'knn', 'dt'], n_select=3)

# Log each model into mlflow separately
for i, model in enumerate(top3, 1):
    with mlflow.start_run(run_name = f"Model: {model}"):
        model_name = "model_" + str(i)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        # Log parameters
        params = model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Predict on the testing set and log metrics
        y_pred = predict_model(model, data=loan_test.drop('Response', axis=1))
        y_test = loan_test['Response']

        # Calculate metrics
        accuracy = sk_metrics.accuracy_score(y_test, y_pred["prediction_label"])
        precision = sk_metrics.precision_score(y_test, y_pred["prediction_label"], average='weighted')
        recall = sk_metrics.recall_score(y_test, y_pred["prediction_label"], average='weighted')
        f1 = sk_metrics.f1_score(y_test, y_pred["prediction_label"], average='weighted')

        # Log metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        mlflow.end_run()


# Split data into training and testing sets
loan_train, loan_test = train_test_split(df, test_size=0.2, random_state=42)

# Define the list of max_depth values to try
max_depth_values = [2, 4, 6, 8, 10]

# Loop over each max_depth value
for depth in max_depth_values:
    with mlflow.start_run(run_name=f"Decision Tree (Max Depth: {depth})"):
        # Initialize and train the decision tree model
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(loan_train.drop('Response', axis=1), loan_train['Response'])

        # Log model parameters
        mlflow.log_param("max_depth", depth)

        # Predict on the testing set and log metrics
        y_pred = model.predict(loan_test.drop('Response', axis=1))
        y_test = loan_test['Response']

        # Calculate metrics
        accuracy = sk_metrics.accuracy_score(y_test, y_pred)
        precision = sk_metrics.precision_score(y_test, y_pred, average='weighted')
        recall = sk_metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = sk_metrics.f1_score(y_test, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        # Log the trained model
        mlflow.sklearn.log_model(model, "decision_tree_model")

        mlflow.end_run()

from sklearn.neighbors import KNeighborsClassifier

# Split data into training and testing sets
loan_train, loan_test = train_test_split(df, test_size=0.2, random_state=42)

# Define the list of n_neighbors values to try
n_neighbors_values = [3, 5, 7, 9, 11]

# Loop over each n_neighbors value
for n_neighbors in n_neighbors_values:
    with mlflow.start_run(run_name=f"KNN (n_neighbors: {n_neighbors})"):
        # Initialize and train the KNN model
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(loan_train.drop('Response', axis=1), loan_train['Response'])

        # Log model parameters
        mlflow.log_param("n_neighbors", n_neighbors)

        # Predict on the testing set and log metrics
        y_pred = model.predict(loan_test.drop('Response', axis=1))
        y_test = loan_test['Response']

        # Calculate metrics
        accuracy = sk_metrics.accuracy_score(y_test, y_pred)
        precision = sk_metrics.precision_score(y_test, y_pred, average='weighted')
        recall = sk_metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = sk_metrics.f1_score(y_test, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        # Log the trained model
        mlflow.sklearn.log_model(model, "knn_model")

        mlflow.end_run()