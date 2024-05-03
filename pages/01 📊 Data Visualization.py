import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url, output_format="PNG", width=300)

st.title("Data Visualization")

df_unclean = pd.read_csv("ifood-data.csv")

st.dataframe(df_unclean)

st.header("Cleaning the Data")
st.metric(value = df_unclean.shape[0], label = "Rows")
st.write('After dropping NA and cleaning up outliers')
df = df_unclean.dropna()
df = df[df["Year_Birth"] > 1940]

st.metric(value = df.shape[0], label = "Rows")

st.metric(value = df_unclean.shape[0] - df.shape[0], label = "Difference")

# Education Levels

st.header("Education Distribution")
st.bar_chart(df.groupby("Education").size(), color = "#FF3008")

st.header("Birth Year Distribution")
st.bar_chart(df.groupby("Year_Birth").size(), color = "#FF3008")

st.header("Accepted Campaigns Distribution")
accepted_cmp_dataset = df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"]]
counts = accepted_cmp_dataset.sum()
counts = counts.reset_index()
counts.columns = ['Campaign', 'Frequency']
st.bar_chart(counts.set_index('Campaign'), color = "#FF3008")

df['Education'] = df['Education'].astype('category').cat.codes
df['Marital_Status'] = df['Marital_Status'].astype('category').cat.codes

df = df.drop(["Dt_Customer"], axis = 1)
df = df.drop(["ID"], axis = 1)

st.header("Heatmap")
dflimed = df["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Recency"]
heatmap = plt.figure(figsize=(18, 10))
sns.heatmap(dflimed.corr().round(2), annot=True, cmap="Reds")
st.pyplot(heatmap)

# limit to just the most correlated vars

recency_response = plt.figure()
sns.boxplot(x=df['Response'], y=df['Recency'], color = "#ff3008")
plt.xlabel('Response')
plt.ylabel('Recency')
plt.title('Box Plot of Response vs Recency')
st.pyplot(recency_response)

response_income = plt.figure()
sns.barplot(x=df['Response'], y=df['Income'], color = "#ff3008")
plt.xlabel('Response')
plt.ylabel('Income')
plt.title('Bar Plot of Response vs Income')
st.pyplot(response_income)

response_teenhome = plt.figure()
sns.boxplot(x=df['Response'], y=df['Teenhome'], color = "#ff3008")
plt.xlabel('Response')
plt.ylabel('Teenhome')
plt.title('Box Plot of Response vs Teenhome')
st.pyplot(response_teenhome)

response_kidhome = plt.figure()
sns.boxplot(x=df['Response'], y=df['Kidhome'], color = "#ff3008")
plt.xlabel('Response')
plt.ylabel('Kidhome')
plt.title('Box Plot of Response vs Kidhome')
st.pyplot(response_kidhome)
