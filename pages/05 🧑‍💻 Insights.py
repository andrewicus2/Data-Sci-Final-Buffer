import streamlit as st
import pandas as pd
from PIL import Image

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url,  output_format="PNG", width=300)

df_unclean = pd.read_csv("ifood-data.csv")
df = df_unclean.dropna()
df = df[df["Year_Birth"] > 1940]


st.title("Insights")

st.header("Our Ideal Customer")

response_by_age = df.groupby('Year_Birth')['Response'].mean()

# Finding the age group with the highest proportion of Response equal to 1
ideal_age = response_by_age.idxmax()
highest_response_proportion = response_by_age.max()

st.metric(value = ideal_age, label = "Ideal Birth Year")
