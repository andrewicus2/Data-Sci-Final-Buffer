import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url, output_format="PNG", width=300)

st.title("MLflow Dashboard")

img_path = 'imgs/ml-flow1.jpeg'
st.image(img_path, caption='ML Flow Image')