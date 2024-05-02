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
from sklearn.metrics import accuracy_score
import time
from shapash.explainer.smart_explainer import SmartExplainer

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
        model = LogisticRegression()
    elif(model == "K-Nearest Neighbors"):
        model = KNeighborsClassifier()
    else:
        model = DecisionTreeClassifier(max_depth=3)


    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    model_accuracy = metrics.accuracy_score(y_test, y_pred)

    model_end_time = time.time()
    model_execution_time = model_end_time - model_start_time

    emissions = tracker.stop()

    # Compile SmartExplainer
    xpl = SmartExplainer(model)
    y_pred = pd.Series(y_pred)
    X_test = X_test.reset_index(drop=True)
    xpl.compile(x=X_test, y_pred=y_pred)

    st.header("Key Metrics")

    col1, col2, col3 = st.columns(3)

    # Metric 1: Accuracy
    col1.metric(label="Accuracy", value=str(round(model_accuracy*100, 2)) + "%")

    # Metric 2: Execution time
    col2.metric(label="Execution time", value=str(round(model_execution_time, 2)) + "s")

    # Metric 3: CO2 Emissions
    col3.metric(label="CO2 Emissions", value=str(round(emissions, 2)) + "kg")

    st.header("Explainable AI")
    st.plotly_chart(xpl.plot.features_importance(), use_container_width = True)

    import random
    subset = random.choices(X_test.index, k =50)
    st.plotly_chart(xpl.plot.features_importance(selection=subset), use_container_width = True)

    st.plotly_chart(xpl.plot.contribution_plot(params[0]), use_container_width = True)

    if(model == "Decision Tree"):
        import graphviz
        from sklearn.tree import export_graphviz
        # Your code for exporting the decision tree graph
        feature_names = X.columns
        feature_cols = X.columns
        dot_data = export_graphviz(model, out_file=None,
                                feature_names=feature_cols,
                                class_names=['0', '1'],
                                filled=True, rounded=True,
                                special_characters=True)

        # Display the graph using streamlit_graphviz
        st.graphviz_chart(dot_data)

