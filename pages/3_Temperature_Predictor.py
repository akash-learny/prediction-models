import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Temperature Predictor",
    page_icon="",
    layout="wide",
)

# =========================
# LOAD OR TRAIN REVERSE MODEL
# =========================
@st.cache_resource
def load_or_train_reverse_model():
    """Load or train a model that predicts temperatures from doneness score"""
    try:
        reverse_model_path = os.path.join('model', 'reverse_temperature_model.pkl')
        
        # Try to load existing reverse model
        if os.path.exists(reverse_model_path):
            reverse_model = joblib.load(reverse_model_path)
            print("[INFO] Reverse model loaded successfully")
            return reverse_model, "loaded"
        
        # If not exists, train a new reverse model
        print("[INFO] Training new reverse model...")
        train_data_path = os.path.join('model', 'train_data.csv')
        
        if not os.path.exists(train_data_path):
            return None, "no_data"
        
        # Load training data
        df = pd.read_csv(train_data_path)
        
        # Prepare data: X = Doneness_Score, y = Temperature values
        X = df[['Doneness_Score']].values
        y = df[['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 
                'T1--Temp_04_InnerWall', 'T1--Temp_05_Ambient_Temp']].values
        
        # Train multi-output regression model
        reverse_model = MultiOutputRegressor(LinearRegression())
        reverse_model.fit(X, y)
        
        # Save the model
        joblib.dump(reverse_model, reverse_model_path)
        print("[INFO] Reverse model trained and saved successfully")
        
        return reverse_model, "trained"
        
    except Exception as e:
        print(f"[ERROR] Error with reverse model: {e}")
        return None, "error"

reverse_model, model_status = load_or_train_reverse_model()

# =========================
# TITLE
# =========================
st.markdown("""
# Temperature Predictor
<div style="font-size:18px; color:gray; margin-top:-10px;">
    <b> Enter a doneness score to predict temperature sensor values </b> <br>
    <br>
    <br>
</div>
""", unsafe_allow_html=True)

# Check if model is available
if reverse_model is None:
    st.error("❌ Unable to load or train the reverse prediction model")
    if model_status == "no_data":
        st.info("Please ensure training data exists at: `model/train_data.csv`")
    st.stop()

# Show model status
if model_status == "trained":
    st.success("✅ Reverse model trained successfully for this session")
elif model_status == "loaded":
    st.success("✅ Reverse model loaded successfully")

# =========================
# INPUT SECTION
# =========================
st.markdown("## Doneness Score Input")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    doneness_score = st.number_input(
        "Doneness Score",
        min_value=0.0,
        max_value=2.5,
        value=1.0,
        step=0.01,
        help="Enter the desired doneness score (typically 0.0 to 2.0)"
    )
    
    # Visual indicator
    st.markdown("### Doneness Level")
    
    if doneness_score < 0.5:
        level = "🟢 Very Low - Just Started"
        color = "#28a745"
    elif doneness_score < 1.0:
        level = "🟡 Low - Early Stage"
        color = "#ffc107"
    elif doneness_score < 1.5:
        level = "🟠 Medium - Cooking"
        color = "#fd7e14"
    else:
        level = "🔴 High - Nearly Done"
        color = "#dc3545"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: {color}20; border-radius: 10px; border-left: 5px solid {color};">
        <h3 style="color: {color}; margin: 0;">{level}</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================
# PREDICTION BUTTON
# =========================
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button("Predict Temperatures", type="primary", use_container_width=True)

# =========================
# PREDICTION LOGIC
# =========================
if predict_button:
    print(f"[INFO] Temperature prediction requested for doneness score: {doneness_score}")
    
    # Prepare input
    X_input = np.array([[doneness_score]])
    
    # Make prediction
    predictions = reverse_model.predict(X_input)[0]
    
    # Extract individual temperature predictions
    temp_lid = predictions[0]
    temp_sidewall = predictions[1]
    temp_handle = predictions[2]
    temp_innerwall = predictions[3]
    temp_ambient = predictions[4]
    
    print(f"[INFO] Predicted temperatures - Lid: {temp_lid:.2f}, SideWall: {temp_sidewall:.2f}, Handle: {temp_handle:.2f}, InnerWall: {temp_innerwall:.2f}, Ambient: {temp_ambient:.2f}")
    
    # Store in session state
    st.session_state['last_temp_prediction'] = {
        'doneness_score': doneness_score,
        'temperatures': {
            'T1--Temp_01_Lid': temp_lid,
            'T1--Temp_02_SideWall': temp_sidewall,
            'T1--Temp_03_Handle': temp_handle,
            'T1--Temp_04_InnerWall': temp_innerwall,
            'T1--Temp_05_Ambient_Temp': temp_ambient
        }
    }
    
    st.markdown("---")
    
    # =========================
    # DISPLAY RESULTS
    # =========================
    st.markdown("##  Predicted Temperature Values")
    
    # Create metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🔵 Lid Temp",
            value=f"{temp_lid:.2f}°C",
            help="Predicted temperature at the lid sensor"
        )
    
    with col2:
        st.metric(
            label="🟢 SideWall Temp",
            value=f"{temp_sidewall:.2f}°C",
            help="Predicted temperature at the side wall sensor"
        )
    
    with col3:
        st.metric(
            label="🟡 Handle Temp",
            value=f"{temp_handle:.2f}°C",
            help="Predicted temperature at the handle sensor"
        )
    
    with col4:
        st.metric(
            label="🟠 InnerWall Temp",
            value=f"{temp_innerwall:.2f}°C",
            help="Predicted temperature at the inner wall sensor"
        )
    
    with col5:
        st.metric(
            label="🟣 Ambient Temp",
            value=f"{temp_ambient:.2f}°C",
            help="Predicted ambient temperature"
        )
    
    st.markdown("---")
    
    # =========================
    # DETAILED ANALYSIS
    # =========================
    st.markdown("### Detailed Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_temp = (temp_lid + temp_sidewall + temp_handle + temp_innerwall) / 4
        st.metric("Average Temp", f"{avg_temp:.2f}°C")
    
    with col2:
        max_temp = max(temp_lid, temp_sidewall, temp_handle, temp_innerwall)
        st.metric("Max Temp", f"{max_temp:.2f}°C")
    
    with col3:
        min_temp = min(temp_lid, temp_sidewall, temp_handle, temp_innerwall)
        st.metric("Min Temp", f"{min_temp:.2f}°C")
    
    with col4:
        temp_range = max_temp - min_temp
        st.metric("Temp Range", f"{temp_range:.2f}°C")
    
    # =========================
    # TEMPERATURE DIFFERENCES
    # =========================
    st.markdown("### Temperature Differences from Ambient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lid_diff = temp_lid - temp_ambient
        st.metric("Lid - Ambient", f"{lid_diff:.2f}°C", delta=f"{lid_diff:.2f}°C")
    
    with col2:
        handle_diff = temp_handle - temp_ambient
        st.metric("Handle - Ambient", f"{handle_diff:.2f}°C", delta=f"{handle_diff:.2f}°C")
    
    with col3:
        innerwall_diff = temp_innerwall - temp_ambient
        st.metric("InnerWall - Ambient", f"{innerwall_diff:.2f}°C", delta=f"{innerwall_diff:.2f}°C")
    
    # =========================
    # PREDICTION SUMMARY TABLE
    # =========================
    with st.expander("📋 Prediction Summary Table"):
        summary_df = pd.DataFrame({
            'Sensor': ['Lid', 'SideWall', 'Handle', 'InnerWall', 'Ambient'],
            'Temperature (°C)': [temp_lid, temp_sidewall, temp_handle, temp_innerwall, temp_ambient],
            'Diff from Ambient (°C)': [
                temp_lid - temp_ambient,
                temp_sidewall - temp_ambient,
                temp_handle - temp_ambient,
                temp_innerwall - temp_ambient,
                0.0
            ]
        })
        st.dataframe(summary_df, use_container_width=True)
    
    # =========================
    # VISUAL TEMPERATURE BAR
    # =========================
    with st.expander("Temperature Visualization"):
        # Create a simple bar chart
        chart_data = pd.DataFrame({
            'Sensor': ['Lid', 'SideWall', 'Handle', 'InnerWall', 'Ambient'],
            'Temperature': [temp_lid, temp_sidewall, temp_handle, temp_innerwall, temp_ambient]
        })
        st.bar_chart(chart_data.set_index('Sensor'))

# =========================
# DISPLAY LAST PREDICTION
# =========================
if 'last_temp_prediction' in st.session_state and not predict_button:
    st.markdown("---")
    last_pred = st.session_state['last_temp_prediction']
    st.info(f"💡 Last prediction for doneness score **{last_pred['doneness_score']:.2f}**: "
            f"Lid={last_pred['temperatures']['T1--Temp_01_Lid']:.2f}°C, "
            f"SideWall={last_pred['temperatures']['T1--Temp_02_SideWall']:.2f}°C, "
            f"Handle={last_pred['temperatures']['T1--Temp_03_Handle']:.2f}°C, "
            f"InnerWall={last_pred['temperatures']['T1--Temp_04_InnerWall']:.2f}°C, "
            f"Ambient={last_pred['temperatures']['T1--Temp_05_Ambient_Temp']:.2f}°C")

# =========================
# QUICK EXAMPLES
# =========================
st.markdown("---")
st.markdown("## Quick Examples")

st.markdown("""
Use these example doneness scores to see typical temperature patterns:
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🟢 Score: 0.25")
    st.markdown("**Just Started**")
    st.code("Very low temps\n~31-32°C range")

with col2:
    st.markdown("### 🟡 Score: 0.75")
    st.markdown("**Early Cooking**")
    st.code("Rising temps\n~38-42°C range")

with col3:
    st.markdown("### 🟠 Score: 1.25")
    st.markdown("**Mid Cooking**")
    st.code("Higher temps\n~48-55°C range")

with col4:
    st.markdown("### 🔴 Score: 1.75")
    st.markdown("**Nearly Done**")
    st.code("Peak temps\n~58-65°C range")
