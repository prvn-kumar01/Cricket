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
@st.cache_resource
def load_model():
    """Loads the saved model and column list using a robust path."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        model_path = os.path.join(root_dir, 'notebook', 'xgb_model.pkl')
        cols_path = os.path.join(root_dir, 'notebook', 'model_columns.pkl')
        
        model = joblib.load(model_path)
        model_cols = joblib.load(cols_path)
        return model, model_cols
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'xgb_model.pkl' and 'model_columns.pkl' are in the notebook directory.")
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

    roles = ['Batsman', 'All-Rounder', 'Bowler', 'Wicket Keeper']
    teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions', 
             'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Gujarat Titans',
             'Pune Warriors India', 'Rajasthan Royals', 'Delhi Daredevils',
             'Chennai Super Kings', 'Rising Pune Supergiant', 'Delhi Capitals',
             'Lucknow Super Giants', 'Punjab Kings']
    
    # **THE MAIN FIX IS HERE**
    # The list now exactly matches your dataset's categories.
    origins = ['Indian', 'Overseas']

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_role = st.selectbox('**Player Role**', sorted(roles))
    with col2:
        selected_team = st.selectbox('**Last Known Team**', sorted(teams))
    with col3:
        selected_origin = st.selectbox('**Player Origin**', sorted(origins))

    if st.button('**Predict Auction Price**', type="primary", use_container_width=True):
        input_data = pd.DataFrame(columns=model_cols)
        input_data.loc[0] = 0
        
        role_col = f'Role_{selected_role}'
        team_col = f'Team_{selected_team}'
        origin_col = f'Player Origin_{selected_origin}'

        # The model only knows the columns it was trained on.
        # This logic checks if the user's selection is valid.
        unsupported_options = []
        if role_col not in input_data.columns:
            unsupported_options.append(selected_role)
        if team_col not in input_data.columns:
            unsupported_options.append(selected_team)
        if origin_col not in input_data.columns:
            unsupported_options.append(selected_origin)

        if unsupported_options:
            st.error(f"Model doesn't support these options: {', '.join(unsupported_options)}")
        else:
            # If all selections are valid, proceed with the prediction
            input_data.loc[0, role_col] = 1
            input_data.loc[0, team_col] = 1
            input_data.loc[0, origin_col] = 1

            prediction = model.predict(input_data)[0]
            predicted_price_crores = prediction / 1_00_00_000

            st.success(f'### Predicted Price: ₹ {predicted_price_crores:.2f} Crores')
            st.balloons()