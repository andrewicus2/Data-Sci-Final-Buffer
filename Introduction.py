import streamlit as st
import pandas as pd
from PIL import Image

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url,  output_format="PNG", width=300)

st.title("Doordash CRM Data Analysis")

st.markdown("##### Context")
st.markdown("Doordash is one of the leading food delivery apps in the United States, present in over seven thousand cities.")
st.markdown("Keeping a high customer engagement is key for growing and consolidating the company’s position as the market leader.")
st.markdown("To expand their product offering, the company is currently looking to launch a physical button that customers can press to automatically place an order for their favorite meal. Doordash is looking to maximize marketing efforts for this new product.")

st.image('amazon-dash.png', caption="DoorDash's new Insta-Order Button", width = 300)

st.markdown("##### Objectives")
st.markdown("The objective of the team is to build a predictive model that will produce the highest profit for the next direct marketing campaign, scheduled for next month. The new campaign, sixth, aims at selling a new gadget to the Customer Database. The team is set on developing a model that predicts customer response to various marketing tactics, which will then be applied to the rest of the customer base.")
st.markdown("Hopefully the model will allow the company to cherry pick the customers that are most likely to purchase the offer while leaving out the non-respondents, making the sixth campaign highly profitable. Moreover, other than maximizing the profit of this new campaign while reducing expenses, the CMO is interested in understanding the characteristic features of those customers who are responsive to purchasing the gadget.")

st.markdown("### Key Goals:")
st.markdown("1. Propose and describe a customer segmentation based on customers behaviors.")
st.markdown("2. Create a predictive model which allows the company to maximize the profits while reducing expenses of the sixth marketing campaign.")
st.markdown("3. By examining which past campaigns were the most responsive, the team can implement the most successful strategies into the sixth campaign to increase customer retention.")

st.markdown("##### Data Source")
st.markdown("The data set contains socio-demographic and firmographic features from about 2.240 customers who were contacted. Additionally, it contains a flag for those customers who responded to the campaign by purchasing the product.")

df = pd.read_csv("ifood-data.csv")

num = st.number_input('No. of Rows', 5, 10)

head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
if head == 'Head':
  st.dataframe(df.head(num))
else:
  st.dataframe(df.tail(num))

st.text('(Rows,Columns)')
st.write(df.shape)

st.markdown("### Fields")
st.markdown("- AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise")
st.markdown("- AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise")
st.markdown("- AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise")
st.markdown("- AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise")
st.markdown("- AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise")
st.markdown("- Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise")
st.markdown("- Complain - 1 if customer complained in the last 2 years")
st.markdown("- DtCustomer - date of customer’s enrolment with the company")
st.markdown("- Education - customer’s level of education")
st.markdown("- Marital - customer’s marital status")
st.markdown("- Kidhome - number of small children in customer’s household")
st.markdown("- Teenhome - number of teenagers in customer’s household")
st.markdown("- Income - customer’s yearly household income")
st.markdown("- MntFishProducts - amount spent on fish products in the last 2 years")
st.markdown("- MntMeatProducts - amount spent on meat products in the last 2 years")
st.markdown("- MntFruits - amount spent on fruits products in the last 2 years")
st.markdown("- MntSweetProducts - amount spent on sweet products in the last 2 years")
st.markdown("- MntWines - amount spent on wine products in the last 2 years")
st.markdown("- MntGoldProds - amount spent on gold products in the last 2 years")
st.markdown("- NumDealsPurchases - number of purchases made with discount")
st.markdown("- NumCatalogPurchases - number of purchases made using catalogue")
st.markdown("- NumStorePurchases - number of purchases made directly in stores")
st.markdown("- NumWebPurchases - number of purchases made through company’s web site")
st.markdown("- NumWebVisitsMonth - number of visits to company’s web site in the last month")
st.markdown("- Recency - number of days since the last purchase")


st.markdown("### Description of Data")
st.dataframe(df.describe())

st.markdown("### Missing Values")
st.markdown("Null or NaN values.")

dfnull = df.isnull().sum()/len(df)*100
totalmiss = dfnull.sum().round(2)
totalmiss = round(totalmiss/len(df.columns),2)
st.write("Percentage of total missing values: ",totalmiss)
st.write(dfnull)
if totalmiss <= 30:
    st.success("We have less then 30 percent of missing values, which is good. This provides us with more accurate data as the null values will not significantly affect the outcomes of our conclusions. And no bias will steer towards misleading results. ")
else:
    st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

st.markdown("### Completeness")
st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

st.write("Total data length:", len(df))
nonmissing = (df.notnull().sum().round(2))
completeness= round(sum(nonmissing)/df.size,2)
st.write("Completeness ratio:",completeness)
st.write(nonmissing)
if completeness >= 0.80:
    st.success("We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")
else:
    st.success("Poor data quality due to low completeness ratio( less than 0.85).")