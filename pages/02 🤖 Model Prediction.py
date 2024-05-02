import streamlit as st
import pandas as pd
from PIL import Image
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from codecarbon import EmissionsTracker
import time

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url,  output_format="PNG", width=300)

st.title("Model Prediction")

df_unclean = pd.read_csv("ifood-data.csv")
df = df_unclean.dropna()
df = df[df["Year_Birth"] > 1940]
df['Education'] = df['Education'].astype('category').cat.codes
df['Marital_Status'] = df['Marital_Status'].astype('category').cat.codes
df = df.drop(["Dt_Customer"], axis = 1)
df = df.drop(["ID"], axis = 1)
params = st.multiselect("Select Parameters", df.columns, default = ["Year_Birth"])
model = st.selectbox("Select Model", ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree"])

if not params:
    st.warning("Please select at least one parameter.")
else:

    X = df.drop(labels = ['Response'], axis = 1)
    X = df[params]
    y = df["Response"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
    model_start_time = time.time()
    tracker = EmissionsTracker()
    tracker.start()
    if(model == "Logistic Regression"):
        logmodel = LogisticRegression()
        logmodel.fit(X_train,y_train)
        model_accuracy = logmodel.predict(X_test)
    elif(model == "K-Nearest Neighbors"):

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        model_accuracy = knn.predict(X_test)
    else:
        clf = DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(X_train,y_train)
        model_accuracy = clf.predict(X_test)

        import graphviz
        from sklearn.tree import export_graphviz

        # Assuming `clf` and `X` are defined somewhere in your code

        # Your code for exporting the decision tree graph
        feature_names = X.columns
        feature_cols = X.columns
        dot_data = export_graphviz(clf, out_file=None,
                                feature_names=feature_cols,
                                class_names=['0', '1'],
                                filled=True, rounded=True,
                                special_characters=True)

        # Display the graph using streamlit_graphviz
        st.graphviz_chart(dot_data)

    model_end_time = time.time()
    model_execution_time = model_end_time - model_start_time


    emissions = tracker.stop()
    print(f"Estimated emissions for training the model: {emissions:.4f} kg of CO2")

    st.metric(label = "Accuracy", value = str(round(metrics.accuracy_score(y_test, model_accuracy)*100, 2)) + "%")
    st.metric(label = "Execution time:", value = str(model_execution_time + "s"))
