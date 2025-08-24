import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

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
        # For example: /.../CrickSight/src/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go one level up to the root directory (e.g., /.../CrickSight/)
        root_dir = os.path.dirname(script_dir)

        # Create the full path to the model files in the notebook directory
        model_path = os.path.join(root_dir, 'notebook', 'xgb_model.pkl')
        cols_path = os.path.join(root_dir, 'notebook', 'model_columns.pkl')
        
        # Load the files using their full paths
        model = joblib.load(model_path)
        model_cols = joblib.load(cols_path)
        
        # Validate that model_cols is a list or array
        if not isinstance(model_cols, (list, pd.Index, np.ndarray)):
            st.error("Invalid model columns format")
            return None, None
            
        return model, model_cols
    except FileNotFoundError as e:
        st.error(f"Model files not found! Please ensure 'xgb_model.pkl' and 'model_columns.pkl' are in the notebook directory. Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
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
if model is not None and model_cols is not None:
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
        try:
            # Validate that model_cols is available
            if model_cols is None or len(model_cols) == 0:
                st.error("Model columns not properly loaded")
                st.stop()
                
            input_data = pd.DataFrame(columns=model_cols)
            input_data.loc[0] = 0

            role_col = f'Role_{selected_role}'
            team_col = f'Team_{selected_team}'
            country_col = f'Country_{selected_country}'

            # Validate that the columns exist in the model
            missing_columns = []
            if role_col not in input_data.columns:
                missing_columns.append(role_col)
            if team_col not in input_data.columns:
                missing_columns.append(team_col)
            if country_col not in input_data.columns:
                missing_columns.append(country_col)
                
            if missing_columns:
                st.error(f"Model doesn't support these options: {', '.join(missing_columns)}")
                st.stop()

            # Set the selected values
            input_data.loc[0, role_col] = 1
            input_data.loc[0, team_col] = 1
            input_data.loc[0, country_col] = 1

            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Validate prediction
            if prediction is None or prediction < 0:
                st.error("Invalid prediction result")
                st.stop()
                
            # Convert to crores (1 crore = 10,000,000)
            predicted_price_crores = prediction / 1_00_00_000

            st.success(f'### Predicted Price: ₹ {predicted_price_crores:.2f} Crores')
            st.balloons()
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
else:
    st.error("Unable to load the model. Please check if the model files exist in the notebook directory.")