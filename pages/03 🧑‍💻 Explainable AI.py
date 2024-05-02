import streamlit as st
import pandas as pd
from PIL import Image
from shapash.explainer.smart_explainer import SmartExplainer

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url,  output_format="PNG", width=300)

st.title("Explainable AI")

xpl = SmartExplainer(clf)

y_pred = pd.Series(y_pred)
X_test = X_test.reset_index(drop=True)
xpl.compile(x=X_test, y_pred=y_pred)


