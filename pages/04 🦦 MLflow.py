import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url, output_format="PNG", width=300)

st.title("MLflow Dashboard")

for i in range(1, 5):
    st.image(str('imgs/ml-flow'+i+'.jpeg'), caption='ML Flow Image')