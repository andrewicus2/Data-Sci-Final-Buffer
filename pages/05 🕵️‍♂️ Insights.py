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

row1col1, row1col2, row1col3 = st.columns(3)
row2col1, row2col2, row2col3 = st.columns(3)

row1col1.metric(value = ideal_age, label = "Ideal Birth Year")

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

row1col2.metric(value = str(round(most_responsive_recency_avg, 2)), label = "Days since last purchase")

df['NumWebVisitsMonth_bins'] = pd.cut(df['NumWebVisitsMonth'], bins=num_bins)

average_response = df.groupby('NumWebVisitsMonth')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_NumWebVisitsMonth_avg = df.loc[df['NumWebVisitsMonth'] == most_responsive_range, 'NumWebVisitsMonth'].mean()

row1col3.metric(value = str(round(most_responsive_NumWebVisitsMonth_avg, 2)), label = "Web visits per month")

df['MntWines_bins'] = pd.cut(df['MntWines'], bins=num_bins)

average_response = df.groupby('MntWines')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_MntWines_avg = df.loc[df['MntWines'] == most_responsive_range, 'MntWines'].mean()

row2col1.metric(value = "$"+ str(round(most_responsive_MntWines_avg, 2)), label = "Amount spent on wine")

df['MntGoldProds_bins'] = pd.cut(df['MntGoldProds'], bins=num_bins)

average_response = df.groupby('MntGoldProds_bins')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_MntGoldProds_avg = df.loc[df['MntGoldProds_bins'] == most_responsive_range, 'MntGoldProds'].mean()

row2col2.metric(value = "$"+ str(round(most_responsive_MntGoldProds_avg, 2)), label = "Amount spent on gold")

df['NumCatalogPurchases_bins'] = pd.cut(df['NumCatalogPurchases'], bins=num_bins)

average_response = df.groupby('NumCatalogPurchases')['Response'].mean()

most_responsive_range = average_response.idxmax()

most_responsive_NumCatalogPurchases_avg = df.loc[df['NumCatalogPurchases'] == most_responsive_range, 'NumCatalogPurchases'].mean()

row2col3.metric(value = str(round(most_responsive_NumCatalogPurchases_avg, 2)), label = "Num of catalog purchases")

st.header("Insights")

st.write("""
The most important variable is recency, symbolizing Doordash’s success in customer loyalty. It suggests that once a customer makes a purchase using the app, they will likely engage in our marketing campaigns and make repeated purchases. By targeting customers who recently made a purchase in our sixth campaign, we can continue to profit from their repeated purchasing behavior. These customers are more likely to be willing to pay for the DoorDash button, as they are likely to experience high satisfaction from the added convenience of ordering food. For the future, we will explore ways of how to expand our loyal customer base. 
""")

st.write("""
The top 2, 3, and 4 variables pertain to whether the client accepted CMP5, CMP 3, and CMP 1. Therefore, for our sixth marketing campaign, we will adopt the same marketing strategies employed in these campaigns to achieve a high retention rate with our customer base. We will also examine the shortcomings of CMP2 and CMP 4, which did not yield a high response rate, to ensure that we avoid the same mistake in our sixth marketing campaign. 
""")

st.write("""
The top 5 and 6 variables are linked to customer spending on food consumption; MntWines and MntMeatProducts. Customers that spent a considerable amount on wine and meat products in the last two years are likely to derive high satisfaction from the Doordash button, as it provides added convenience in obtaining these products. Moreover, given that wine is often considered a luxury rather than a necessity, we can infer that customers who have purchased a significant amount of wine are more inclined to invest in food experiences and thus, will likely invest in the button.
""")

st.write("""
Information on the customer’s family, including marital status and whether they had a child or teenager at home, has a negligible effect on whether the customer responded to the marketing campaign. Unless this changes in the future, we do not need to expend resources gathering or analyzing this information. 
""")

st.header("Conclusion")

st.write("""
While this dataset gives us good insight into the correlation between whether the customer responded to past marketing campaigns and how they respond to the current one, it provides no information about the content of each marketing campaign. As a result, our analysis is entirely based on the recency of each campaign. This poses a significant limitation to our understanding of the data because the response rates to each campaign may have also been influenced by qualitative differences that we do not have access to. 
""")