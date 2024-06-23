import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="Society Recommender", page_icon="✅", layout="centered")

# Load the data
df = pd.read_csv('datasets/data_cleaned.csv')
df['address'] = df['society'] + ', ' + df['location'] + ', ' + df['city']

# Load user input and predicted price
r_df = pd.read_csv('datasets/recommend_df.csv')

# Load socities link
l_df = pd.read_csv('datasets/data_link.csv')
l_df['society'] = l_df['location'].apply(lambda x: x.split(',')[0])

features = ['education', 'banks', 'shoppings', 'office_complexes']


def recommend_features(education, banks, shoppings, office_complexes, city, price_min, price_max):
    # Filter the DataFrame based on city and initial price range
    filtered_df = df[(df['city'] == city) & (df['price_in_lakh'] >= price_min) & (df['price_in_lakh'] <= price_max)]

    if filtered_df.empty:
        return pd.DataFrame({'Society': [], 'Distance': []})

    # Create an input vector from the user's input
    input_vector = [[education, banks, shoppings, office_complexes]]

    # Prepare the features and target
    features = ['education', 'banks', 'shoppings', 'office_complexes']
    X = filtered_df[features]
    y = filtered_df['society']

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    # Predict the society based on the input vector
    predicted_society = clf.predict(input_vector)

    # Get the predicted society
    predicted_society = predicted_society[0]

    # Get the top 5 societies
    recommended_societies = [predicted_society]
    min_price = price_min
    max_price = price_max
    expand_range = True

    while len(recommended_societies) < 5 and expand_range:
        # Ensure unique societies in recommendations
        for idx in filtered_df.index:
            society = df['society'].iloc[idx]
            location = df['location'].iloc[idx]
            society = society + ', ' + location
            if society not in recommended_societies:
                recommended_societies.append(society)
                if len(recommended_societies) == 5:
                    break

        # Check if we need to expand the price range
        if len(recommended_societies) < 5:
            min_price -= 10
            max_price += 10

            # Check if expanded range is within reasonable limits
            if min_price < 0:
                min_price = 0
            if max_price > df['price_in_lakh'].max():
                max_price = df['price_in_lakh'].max()

            # Filter again with expanded price range
            filtered_df = df[(df['city'] == city) & (df['price_in_lakh'] >= min_price) & (df['price_in_lakh'] <= max_price)]

            if filtered_df.empty:
                expand_range = False

    # Create a DataFrame for recommendations
    recommendations_df = pd.DataFrame({
        'Society': recommended_societies[:5],
        'Distance': [0] * len(recommended_societies[:5])  # Distance is not applicable here
    })

    return recommendations_df

def recommend_properties_generalized(criteria, value, city, price_min, price_max):
    # Initialize variables
    recommended_societies = []
    iterations = 0
    price_step = 10

    # Define the input vector for the user's specified criterion range
    input_vector = [[value]]

    while len(recommended_societies) < 5 and iterations < 10:
        # Filter the DataFrame based on city and current price range
        filtered_df = df[(df['city'] == city) & (df['price_in_lakh'] >= price_min) & (df['price_in_lakh'] <= price_max)]

        if filtered_df.empty:
            # If no properties match the filter, expand the price range and try again
            price_min -= price_step
            price_max += price_step
            iterations += 1
            continue

        # Calculate Manhattan distance between input vector and properties in filtered DataFrame
        filtered_indices = filtered_df.index.tolist()
        distances = pairwise_distances(input_vector, df.loc[filtered_indices, [criteria]], metric='manhattan')[0]

        # Get the sorted indices based on distances (ascending order)
        sorted_indices = np.argsort(distances)

        # Ensure unique societies in recommendations
        for idx in sorted_indices:
            society = df.loc[filtered_indices[idx], 'society']
            location = df.loc[filtered_indices[idx], 'location']
            if society not in [rec[0] for rec in recommended_societies]:
                recommended_societies.append((society, location, distances[idx]))
                if len(recommended_societies) == 5:
                    break

        iterations += 1

    # Create a DataFrame for recommendations
    recommendations_df = pd.DataFrame(recommended_societies, columns=['Society', 'Location', 'Distance'])
    recommendations_df['Society'] = recommendations_df['Society'] + ', ' + recommendations_df['Location']
    recommendations_df.drop(columns=['Location'], inplace=True)

    return recommendations_df

def merge_and_normalize_recommendations(recommend1, recommend2, recommend3):
    # Concatenate all recommendations
    combined_recommendations = pd.concat([recommend1, recommend2, recommend3])

    # Normalize the distances using Min-Max scaling
    scaler = MinMaxScaler()
    combined_recommendations['Normalized_Distance'] = scaler.fit_transform(combined_recommendations[['Distance']])

    # Identify duplicates and retain them in the best recommendations
    duplicates = combined_recommendations[combined_recommendations.duplicated(subset=['Society'], keep=False)]
    unique_duplicates = duplicates.drop_duplicates(subset=['Society'], keep='first')
    combined_recommendations = combined_recommendations.drop_duplicates(subset=['Society'], keep=False)

    # Sort duplicates by normalized distance
    unique_duplicates.sort_values(by='Normalized_Distance', inplace=True)
    best_duplicates = unique_duplicates.head(5)

    # Sort remaining recommendations by normalized distance
    combined_recommendations.sort_values(by='Normalized_Distance', inplace=True)

    # Select top recommendations to make up a total of 5
    remaining_recommendations_needed = 5 - len(best_duplicates)
    best_remaining_recommendations = combined_recommendations.head(remaining_recommendations_needed)

    # Concatenate the best duplicates and best remaining recommendations
    best_recommendations = pd.concat([best_duplicates, best_remaining_recommendations])

    # Drop the 'Metric' and 'Normalized_Distance' columns for the final output
    best_recommendations.drop(columns=['Distance','Normalized_Distance'], inplace=True)

    return best_recommendations

# Creating a list of nearby places
nearby = []
if r_df['education'].get(0):
    nearby.append('Schools, Colleges, and Libraries')
if r_df['banks'].get(0):
    nearby.append('Banks/ATMs')
if r_df['shoppings'].get(0):
    nearby.append('Shopping Malls')
if r_df['office_complexes'].get(0):
    nearby.append('Office Complexes')

# Calculating the base price range
base_price = r_df['price'].get(0)
price_range_low = round(base_price - 15, 2)
price_range_high = round(base_price + 15, 2)

city = r_df['city'].get(0)

# Title and introduction with enhanced output
st.title(f"Welcome to {city}")

# Introduction with bullet points for nearby places
st.markdown(f"""
Hey, as per your input, you want a **{round(r_df['bedroom'].get(0))} BHK** apartment with a **{round(r_df['area_in_sqft'].get(0))} sqft** area, located near:
- **{nearby[0] if nearby else 'No specific places mentioned'}**
""")

# Adding the rest of the nearby places
for place in nearby[1:]:
    st.markdown(f"- **{place}**")

# Predicted price range
st.markdown(f"""
The predicted price is: **{price_range_low} Lac** to **{price_range_high} Lac**
""")

user_input = [r_df['education'].get(0), r_df['banks'].get(0), r_df['shoppings'].get(0), r_df['office_complexes'].get(0)]

if st.button('Recommend'):
    # Provide feedback that the recommendation is being generated
    with st.spinner('Generating recommendations...'):
        recommendation1 = recommend_features(*user_input, city, price_range_low, price_range_high)
        recommendation2 = recommend_properties_generalized('area_in_sqft', r_df['area_in_sqft'].get(0), city, price_range_high, price_range_low)
        recommendation3 = recommend_properties_generalized('bedroom', r_df['bedroom'].get(0), city, price_range_high, price_range_low)

    best_recommendations = merge_and_normalize_recommendations(recommendation1, recommendation2, recommendation3)

    best_recommendations['society'] = best_recommendations['Society'].apply(lambda x: x.split(',')[0])

    final_df = best_recommendations.merge(l_df, how='left', on=['society'])

    # final_df = final_df.merge(df[['address','price_in_lakh','area_in_sqft']], how='left',left_on=['location'],right_on=['address'])
    
    # Display 5 society names with clickable links in a more attractive way
    st.markdown("### Top 5 Recommendations")
    st.write("Here are the top 5 societies that match your preferences:")

    for index, row in final_df.head(5).iterrows():
        st.markdown(f"#### [{row['society']}]({row['link']})")
        st.write(f"Location: {row['location']}")
        # st.write(f"Price Range: {row['price_in_lakh']-20} Lac to {row['price_in_lakh']+20} Lac")
        # st.write(f"Area: {row['area_in_sqft']} sqft")
        st.write("---")
