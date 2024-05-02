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

# Define the number of bins for recency segmentation
num_bins = 10

# Create bins for recency segmentation
df['Recency_bins'] = pd.cut(df['Recency'], bins=num_bins)

# Calculate the average response for each recency bin
average_response = df.groupby('Recency_bins')['Response'].mean()

# Find the recency range with the highest average response
most_responsive_range = average_response.idxmax()

# Find the average rececny within the most responsive range
most_responsive_recency_avg = df.loc[df['Recency_bins'] == most_responsive_range, 'Recency'].mean()

st.metric(value = str(round(most_responsive_recency_avg, 2)), label = "Days since last visit")

df['NumWebVisitsMonth_bins'] = pd.cut(df['NumWebVisitsMonth'], bins=num_bins)

average_response = df.groupby('NumWebVisitsMonth')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_NumWebVisitsMonth_avg = df.loc[df['NumWebVisitsMonth'] == most_responsive_range, 'NumWebVisitsMonth'].mean()

st.metric(value = str(round(most_responsive_NumWebVisitsMonth_avg, 2)), label = "Web visits per month")

df['MntWines_bins'] = pd.cut(df['MntWines'], bins=num_bins)

average_response = df.groupby('MntWines')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_MntWines_avg = df.loc[df['MntWines'] == most_responsive_range, 'MntWines'].mean()

st.metric(value = "$"+ str(round(most_responsive_MntWines_avg, 2)), label = "Amount spent on wine")

df['MntGoldProds_bins'] = pd.cut(df['MntGoldProds'], bins=num_bins)

average_response = df.groupby('MntGoldProds_bins')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_MntGoldProds_avg = df.loc[df['MntGoldProds_bins'] == most_responsive_range, 'MntGoldProds'].mean()

st.metric(value = "$"+ str(round(most_responsive_MntGoldProds_avg, 2)), label = "Amount spent on gold")

df['NumCatalogPurchases_bins'] = pd.cut(df['NumCatalogPurchases'], bins=num_bins)

average_response = df.groupby('NumCatalogPurchases')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_NumCatalogPurchases_avg = df.loc[df['NumCatalogPurchases'] == most_responsive_range, 'NumCatalogPurchases'].mean()

st.metric(value = str(round(most_responsive_NumCatalogPurchases_avg, 2)), label = "Num of catalog purchases")
