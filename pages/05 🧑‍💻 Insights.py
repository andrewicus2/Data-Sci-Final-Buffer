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

import streamlit as st
import pandas as pd

# Define your DataFrame 'df'

# Define the number of bins for recency segmentation
num_bins = 10

# Define the number of rows and columns for the grid
num_rows = 2
num_columns = 3

# Create bins for recency segmentation
df['Recency_bins'] = pd.cut(df['Recency'], bins=num_bins)
df['NumWebVisitsMonth_bins'] = pd.cut(df['NumWebVisitsMonth'], bins=num_bins)
df['MntWines_bins'] = pd.cut(df['MntWines'], bins=num_bins)
df['MntGoldProds_bins'] = pd.cut(df['MntGoldProds'], bins=num_bins)
df['NumCatalogPurchases_bins'] = pd.cut(df['NumCatalogPurchases'], bins=num_bins)

# Group by Recency bins and calculate the average response for each bin
average_response_recency = df.groupby('Recency_bins')['Response'].mean()
most_responsive_range_recency = average_response_recency.idxmax()
most_responsive_recency_avg = df.loc[df['Recency_bins'] == most_responsive_range_recency, 'Recency'].mean()

# Group by NumWebVisitsMonth bins and calculate the average response for each bin
average_response_web_visits = df.groupby('NumWebVisitsMonth_bins')['Response'].mean()
most_responsive_range_web_visits = average_response_web_visits.idxmax()
most_responsive_web_visits_avg = df.loc[df['NumWebVisitsMonth_bins'] == most_responsive_range_web_visits, 'NumWebVisitsMonth'].mean()

# Group by MntWines bins and calculate the average response for each bin
average_response_wines = df.groupby('MntWines_bins')['Response'].mean()
most_responsive_range_wines = average_response_wines.idxmax()
most_responsive_wines_avg = df.loc[df['MntWines_bins'] == most_responsive_range_wines, 'MntWines'].mean()

# Group by MntGoldProds bins and calculate the average response for each bin
average_response_gold_prods = df.groupby('MntGoldProds_bins')['Response'].mean()
most_responsive_range_gold_prods = average_response_gold_prods.idxmax()
most_responsive_gold_prods_avg = df.loc[df['MntGoldProds_bins'] == most_responsive_range_gold_prods, 'MntGoldProds'].mean()

# Group by NumCatalogPurchases bins and calculate the average response for each bin
average_response_catalog_purchases = df.groupby('NumCatalogPurchases_bins')['Response'].mean()
most_responsive_range_catalog_purchases = average_response_catalog_purchases.idxmax()
most_responsive_catalog_purchases_avg = df.loc[df['NumCatalogPurchases_bins'] == most_responsive_range_catalog_purchases, 'NumCatalogPurchases'].mean()

# Organize metrics into a 2x3 grid
with st.container():
    for row in range(num_rows):
        with st.columns(num_columns):
            for col in range(num_columns):
                index = row * num_columns + col
                if index == 0:
                    st.metric(label="Ideal Birth Year", value=most_responsive_recency_avg)
                elif index == 1:
                    st.metric(label="Days since last visit", value=most_responsive_web_visits_avg)
                elif index == 2:
                    st.metric(label="Amount spent on wine", value="$" + str(round(most_responsive_wines_avg, 2)))
                elif index == 3:
                    st.metric(label="Amount spent on gold", value="$" + str(round(most_responsive_gold_prods_avg, 2)))
                elif index == 4:
                    st.metric(label="Num of catalog purchases", value=most_responsive_catalog_purchases_avg)
                # Add more metrics as needed


row1col1, row1col2, row1col3 = st.columns(3)

row2col1, row2col2, row2col3 = st.columns(3)

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

st.write("""
While this dataset gives us good insight into the correlation between whether the customer responded to past marketing campaigns and how they respond to the current one, it provides no information about the content of each marketing campaign. As a result, our analysis is entirely based on the recency of each campaign. This poses a significant limitation to our understanding of the data because the response rates to each campaign may have also been influenced by qualitative differences that we do not have access to. 
""")