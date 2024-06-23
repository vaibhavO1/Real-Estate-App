import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Analysis App", page_icon="🔎", layout="centered")

st.title('Analytics')

# Load data
new_df = pd.read_csv('datasets/data_cleaned.csv')

# Add a select box for city selection
cities = new_df['city'].unique()
selected_city = st.selectbox('Select a city', sorted(cities))

# Filter the DataFrame for the selected city
filtered_df = new_df[new_df['city'] == selected_city][['location','price_in_lakh', 'price_per_sqft', 'area_in_sqft', 'latitude', 'longitude','bedroom']]

# Group by location and calculate mean values
group_df = filtered_df.groupby('location').mean()[['price_in_lakh', 'price_per_sqft', 'area_in_sqft', 'latitude', 'longitude']]
group_df['area_in_sqft'] = round(group_df['area_in_sqft'],2)
group_df['price_per_sqft'] = round(group_df['price_per_sqft'],2)


st.header(f'Location wise Price per Sqft Geomap for {selected_city}')
fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='area_in_sqft',
                        color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                        mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)

st.plotly_chart(fig, use_container_width=True)

st.header('Socities Wordcloud')

society_list = new_df[new_df['city'] == selected_city]['society'].tolist()
society_text = ' '.join(society_list)

wordcloud = WordCloud(width=800, height=800,
                      background_color='black',
                      min_font_size=10).generate(society_text)

# Display the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
st.pyplot(plt.gcf())

st.header('Area Vs Price')

fig1 = px.scatter(new_df[new_df['city'] == selected_city], x="area_in_sqft", y="price_in_lakh", color="bedroom", title="Area Vs Price",template="plotly_dark")

st.plotly_chart(fig1, use_container_width=True)

st.header('BHK Pie Chart')

location_options = filtered_df['location'].unique().tolist()
location_options.insert(0,'overall')

selected_location = st.selectbox('Select Loction', sorted(location_options))

if selected_location == 'overall':

    fig2 = px.pie(filtered_df, names='bedroom',template="seaborn")

    st.plotly_chart(fig2, use_container_width=True)
else:

    fig2 = px.pie(filtered_df[filtered_df['location'] == selected_location], names='bedroom',template="seaborn")

    st.plotly_chart(fig2, use_container_width=True)

st.header('Side by Side BHK price comparison')

if selected_location == 'overall':
    fig3 = px.box(filtered_df[filtered_df['bedroom'] <= 4], x='bedroom', y='price_in_lakh', title='BHK Price Range',color='bedroom', template="seaborn")

else:
    fig3 = px.box(filtered_df[(filtered_df['location'] == selected_location) & (filtered_df['bedroom'] <= 4)], x='bedroom', y='price_in_lakh', title='BHK Price Range',color='bedroom', template="seaborn")
st.plotly_chart(fig3, use_container_width=True)