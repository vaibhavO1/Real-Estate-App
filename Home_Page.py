import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('datasets/data_cleaned.csv')

# Calculate the mean average price and sum of the number of properties per city
temp = df[['city', 'price_in_lakh']]
plot_1 = temp.groupby('city').mean().reset_index()
plot_2 = temp.groupby('city').count().reset_index()

# Set page config
st.set_page_config(page_title="Real Estate App", page_icon="🏡", layout="centered")

# Title and introduction
st.title("🏡 Real Estate App")
st.markdown("""
Welcome to the Real Estate App. Explore data from different metro cities!
""")

# Define a color palette for the cities
colors = plt.cm.tab20(range(len(plot_1)))

# Plotting a bar chart for price_in_lakhs
st.header("Average Property Prices")
fig, ax = plt.subplots(facecolor='black')
ax.set_facecolor('black')
bars = ax.bar(plot_1['city'], plot_1['price_in_lakh'], color=colors, edgecolor='white')
ax.set_xlabel('City', color='white')
ax.set_ylabel('Average Price of Cities in Lakhs', color='white')
ax.set_title('Average Property Prices by City', color='white')
ax.set_xticks(range(len(plot_1['city'])))
ax.set_xticklabels(plot_1['city'], rotation='vertical', color='white')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
st.pyplot(fig)

# Plotting a bar chart for Number of Properties
st.header("Number of Properties")
fig, ax = plt.subplots(facecolor='black')
ax.set_facecolor('black')
bars = ax.bar(plot_2['city'], plot_2['price_in_lakh'], color=colors, edgecolor='white')
ax.set_xlabel('City', color='white')
ax.set_ylabel('Number of Properties', color='white')
ax.set_title('Number of Properties by City', color='white')
ax.set_xticks(range(len(plot_2['city'])))
ax.set_xticklabels(plot_2['city'], rotation='vertical', color='white')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
st.pyplot(fig)

# Footer
st.markdown("""
Data source: [99acres.com](https://www.99acres.com/)
""")
