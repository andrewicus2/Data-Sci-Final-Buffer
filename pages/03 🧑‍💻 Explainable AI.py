import streamlit as st
import pandas as pd
from PIL import Image
from shapash.explainer.smart_explainer import SmartExplainer

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url,  output_format="PNG", width=300)

st.title("Explainable AI")



