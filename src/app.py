import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Setup ---
st.set_page_config(
    page_title="IPL Auction Predictor",
    page_icon="🏏",
    layout="centered"
)

# --- Model Loading ---
# Using a cache to load the model only once for efficiency
@st.cache_resource
def load_model():
    """Loads the saved model and column list."""
    try:
        model = joblib.load('C:/Praveen/DS Projects/CrickSight/notebook/model_columns.pkl')
        model_cols = joblib.load('C:/Praveen/DS Projects/CrickSight/notebook/xgb_model.pkl')
        return model, model_cols
    except FileNotFoundError:
        st.error("Model files not found! Make sure 'xgb_model.pkl' and 'model_columns.pkl' are in the root directory.")
        return None, None

model, model_cols = load_model()

# --- UI Elements ---
st.title('🏏 IPL Auction Price Predictor')
st.write("""
Welcome! This app predicts the auction price for a cricket player.
Please provide the player's details below to get an estimated auction value.
""")

st.divider()

# --- User Input Section ---
if model is not None:
    st.header('Enter Player Details')

    # Pre-defined options for the dropdowns based on our dataset
    roles = ['Batsman', 'All-Rounder', 'Bowler', 'Wicket Keeper']
    teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions', 
             'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Gujarat Titans',
             'Pune Warriors India', 'Rajasthan Royals', 'Delhi Daredevils',
             'Chennai Super Kings', 'Rising Pune Supergiant', 'Delhi Capitals',
             'Lucknow Super Giants', 'Punjab Kings']
    countries = ['Indian', 'South African', 'Australian', 'New Zealander',
                 'West Indian', 'Sri Lankan', 'Bangladeshi', 'English',
                 'Afghan', 'Namibian']

    # Create columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_role = st.selectbox('**Player Role**', sorted(roles))
    with col2:
        selected_team = st.selectbox('**Last Known Team**', sorted(teams))
    with col3:
        selected_country = st.selectbox('**Player Country**', sorted(countries))

    # --- Prediction Button ---
    if st.button('**Predict Auction Price**', type="primary", use_container_width=True):
        # Create an empty dataframe with the model's expected columns
        input_data = pd.DataFrame(columns=model_cols)
        input_data.loc[0] = 0 # Initialize with zeros

        # Construct the column names for the selected inputs
        role_col = f'Role_{selected_role}'
        team_col = f'Team_{selected_team}'
        country_col = f'Country_{selected_country}'

        # Set the corresponding columns to 1
        if role_col in input_data.columns:
            input_data.loc[0, role_col] = 1
        if team_col in input_data.columns:
            input_data.loc[0, team_col] = 1
        if country_col in input_data.columns:
            input_data.loc[0, country_col] = 1

        # Make the prediction
        prediction = model.predict(input_data)[0]
        predicted_price_crores = prediction / 1_00_00_000

        # Display the result
        st.success(f'### Predicted Price: ₹ {predicted_price_crores:.2f} Crores')
        st.balloons()