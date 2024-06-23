import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Price Predictor", page_icon="💸", layout="centered")

# Load datasets
with open('datasets/final_df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('datasets/pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

st.header('Enter your inputs')

city = st.selectbox('City', sorted(df['city'].unique().tolist()))

# Filter locations based on selected city
filtered_locations = df[df['city'] == city]['location'].unique().tolist()
location = st.selectbox('Location', sorted(filtered_locations))

bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedroom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))

balcony = float(st.selectbox('Number of Balconies', sorted(df['balcony'].unique().tolist())))

property_age = st.selectbox('Property Age', sorted(df['age'].unique().tolist()))

# Limit the Built-up Area input
built_up_area = st.number_input('Area in sqft', min_value=0.0, max_value=10000.0, value=800.0)

luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))

# List of items
nearby = ['Schools, Colleges and Libraries', 'Banks/ATMs', 'Shopping Malls', 'Office Complexes']

# Add a subtitle with a smaller font size
st.markdown("<h4 style='font-size:16px;'>NearBy Places</h4>", unsafe_allow_html=True)

# Create multiselect checkbox
selected_items = st.multiselect('Select items', nearby)

# Initialize a dictionary to store the results
results = {item: 0 for item in nearby}

# Update the dictionary based on selected items
for item in selected_items:
    results[item] = 1

edu = results['Schools, Colleges and Libraries']
bank = results['Banks/ATMs']
shop = results['Shopping Malls']
off = results['Office Complexes']

# Add some space before the predict button
st.markdown("<br>", unsafe_allow_html=True)

# Display the predict button in the center
if st.button('Predict'):
    # Form a dataframe
    data = [[city, location, bedrooms, balcony, property_age, built_up_area, edu, bank, shop, off, bathroom, luxury_category]]
    columns = ['city', 'location', 'bedroom', 'balcony', 'age', 'area_in_sqft',
               'education', 'banks', 'shoppings', 'office_complexes', 'bathroom',
               'luxury_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    # Predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 15
    high = base_price + 15

    # Display the results
    if low < 100 and high > 100:
        st.text("The price of the flat is between {} Lac and {} Cr".format(round(low, 2), round(high/100, 2)))
    elif high < 100:
        st.text("The price of the flat is between {} Lac and {} Lac".format(round(low, 2), round(high, 2)))
    elif low > 100:
        st.text("The price of the flat is between {} Cr and {} Cr".format(round(low/100, 2), round(high/100, 2))) 

    one_df['price'] = base_price

    one_df.to_csv('datasets/recommend_df.csv',index=False)