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
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
import time
from shapash.explainer.smart_explainer import SmartExplainer

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url, output_format="PNG", width=300)

st.title("MLflow Dashboard")

st.image('imgs/ml-flow1.jpeg', caption='MLflow Dashbaord')
st.image('imgs/ml-flow2.jpeg', caption='Comparing Models')
st.image('imgs/ml-flow3.jpeg', caption='Models sorted by Accuracy')
st.image('imgs/ml-flow4.jpeg', caption='Winning Model')

st.download_button(label="Download Winning Model", data="meta_yaml_content", file_name="meta.yaml", mime="text/yaml")
