import streamlit as st
import pandas as pd
import joblib
import os # <-- Yeh library path ko handle karegi

# --- Page Setup ---
st.set_page_config(
    page_title="IPL Auction Predictor",
    page_icon="🏏",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the saved model and column list using a robust path."""
    try:
        # Get the absolute path of the directory where app.py is located
        # For example: /.../Cricket/src/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go one level up to the root directory (e.g., /.../Cricket/)
        root_dir = os.path.dirname(script_dir)

        # Create the full path to the model files
        model_path = os.path.join(root_dir, 'xgb_model.pkl')
        cols_path = os.path.join(root_dir, 'model_columns.pkl')
        
        # Load the files using their full paths
        model = joblib.load(model_path)
        model_cols = joblib.load(cols_path)
        return model, model_cols
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'xgb_model.pkl' and 'model_columns.pkl' are in the main project directory.")
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

    # Pre-defined options for the dropdowns
    roles = ['Batsman', 'All-Rounder', 'Bowler', 'Wicket Keeper']
    teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions', 
             'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Gujarat Titans',
             'Pune Warriors India', 'Rajasthan Royals', 'Delhi Daredevils',
             'Chennai Super Kings', 'Rising Pune Supergiant', 'Delhi Capitals',
             'Lucknow Super Giants', 'Punjab Kings']
    countries = ['Indian', 'South African', 'Australian', 'New Zealander',
                 'West Indian', 'Sri Lankan', 'Bangladeshi', 'English',
                 'Afghan', 'Namibian']

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_role = st.selectbox('**Player Role**', sorted(roles))
    with col2:
        selected_team = st.selectbox('**Last Known Team**', sorted(teams))
    with col3:
        selected_country = st.selectbox('**Player Country**', sorted(countries))

    if st.button('**Predict Auction Price**', type="primary", use_container_width=True):
        input_data = pd.DataFrame(columns=model_cols)
        input_data.loc[0] = 0

        role_col = f'Role_{selected_role}'
        team_col = f'Team_{selected_team}'
        country_col = f'Country_{selected_country}'

        if role_col in input_data.columns:
            input_data.loc[0, role_col] = 1
        if team_col in input_data.columns:
            input_data.loc[0, team_col] = 1
        if country_col in input_data.columns:
            input_data.loc[0, country_col] = 1

        prediction = model.predict(input_data)[0]
        predicted_price_crores = prediction / 1_00_00_000

        st.success(f'### Predicted Price: ₹ {predicted_price_crores:.2f} Crores')
        st.balloons()