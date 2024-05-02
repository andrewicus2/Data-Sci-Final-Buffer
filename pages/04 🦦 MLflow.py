import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url, output_format="PNG", width=300)

st.title("MLflow Dashboard")

st.image('imgs/ml-flow1.jpeg', caption='MLflow Dashbaord')
st.image('imgs/ml-flow2.jpeg', caption='Comparing Models')
st.image('imgs/ml-flow3.jpeg', caption='Models sorted by Accuracy')
st.image('imgs/ml-flow4.jpeg', caption='Winning Model')