import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Doneness Predictor",
    page_icon="",
    layout="wide",
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    """Load the trained model and feature names"""
    try:
        model_path = os.path.join('model', 'rice_doneness_model.pkl')
        features_path = os.path.join('model', 'feature_names.pkl')
        metadata_path = os.path.join('model', 'model_metadata.pkl')
        
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        metadata = joblib.load(metadata_path)
        
        print("[INFO] Model loaded successfully for Doneness Predictor")
        return model, feature_names, metadata
    except FileNotFoundError as e:
        print(f"[INFO] No trained model found: {e}")
        return None, None, None
    except Exception as e:
        print(f"[INFO] Error loading model: {e}")
        return None, None, None

model, feature_names, model_metadata = load_model()

# =========================
# FEATURE ENGINEERING FUNCTION
# =========================
def engineer_features(data):
    """Apply the same feature engineering as training"""
    df = pd.DataFrame([data])
    
    # Create engineered features
    df['Temp_Lid_Ambient_Diff'] = df['T1--Temp_01_Lid'] - df['T1--Temp_05_Ambient_Temp']
    df['Temp_Handle_Ambient_Diff'] = df['T1--Temp_03_Handle'] - df['T1--Temp_05_Ambient_Temp']
    df['Temp_InnerWall_Ambient_Diff'] = df['T1--Temp_04_InnerWall'] - df['T1--Temp_05_Ambient_Temp']
    df['Temp_Range'] = df[['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 'T1--Temp_04_InnerWall']].max(axis=1) - df[['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 'T1--Temp_04_InnerWall']].min(axis=1)
    
    return df

# =========================
# TITLE
# =========================
st.markdown("""
# Doneness Predictor
<div style="font-size:18px; color:gray; margin-top:-10px;">
    <b> Enter temperature values to predict rice doneness score </b> <br>
    <br>
    <br>
</div>
""", unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.error("❌ No trained model found")
    st.info("Please train the model first by running: `python model/train_model.py`")
    st.stop()

# =========================
# INPUT SECTION
# =========================
st.markdown("## Temperature Input")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Temperature Sensors")
    
    temp_lid = st.number_input(
        "T1--Temp_01_Lid (°C)",
        min_value=0.0,
        max_value=150.0,
        value=31.7,
        step=0.1,
        help="Temperature reading from the lid sensor"
    )
    
    temp_sidewall = st.number_input(
        "T1--Temp_02_SideWall (°C)",
        min_value=0.0,
        max_value=150.0,
        value=31.3,
        step=0.1,
        help="Temperature reading from the side wall sensor"
    )
    
    temp_handle = st.number_input(
        "T1--Temp_03_Handle (°C)",
        min_value=0.0,
        max_value=150.0,
        value=30.9,
        step=0.1,
        help="Temperature reading from the handle sensor"
    )

with col2:
    st.markdown("### Additional Sensors")
    
    temp_innerwall = st.number_input(
        "T1--Temp_04_InnerWall (°C)",
        min_value=0.0,
        max_value=150.0,
        value=31.4,
        step=0.1,
        help="Temperature reading from the inner wall sensor"
    )
    
    temp_ambient = st.number_input(
        "T1--Temp_05_Ambient_Temp (°C)",
        min_value=0.0,
        max_value=150.0,
        value=31.3,
        step=0.1,
        help="Ambient temperature reading"
    )

st.markdown("---")

# =========================
# PREDICTION BUTTON
# =========================
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button("Predict Doneness Score", type="primary", use_container_width=True)

# =========================
# PREDICTION LOGIC
# =========================
if predict_button:
    print("[INFO] Manual prediction requested")
    print(f"[INFO] Input values - Lid: {temp_lid}, SideWall: {temp_sidewall}, Handle: {temp_handle}, InnerWall: {temp_innerwall}, Ambient: {temp_ambient}")
    
    # Prepare input data
    input_data = {
        'T1--Temp_01_Lid': temp_lid,
        'T1--Temp_02_SideWall': temp_sidewall,
        'T1--Temp_03_Handle': temp_handle,
        'T1--Temp_04_InnerWall': temp_innerwall,
        'T1--Temp_05_Ambient_Temp': temp_ambient
    }
    
    # Engineer features
    df_features = engineer_features(input_data)
    
    # Select features in correct order
    X_input = df_features[feature_names]
    
    print(f"[INFO] Engineered features: {X_input.shape[1]} features")
    
    # Make prediction
    prediction = model.predict(X_input)[0]
    
    print(f"[INFO] Prediction result: {prediction:.4f}")
    
    # Store in session state
    st.session_state['last_prediction'] = prediction
    st.session_state['last_input'] = input_data
    
    st.markdown("---")
    
    # =========================
    # DISPLAY RESULTS
    # =========================
    st.markdown("## Prediction Result")
    
    # Large display of prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;">
            <h1 style="color: #1f77b4; font-size: 60px; margin: 0;">{prediction:.4f}</h1>
            <p style="font-size: 24px; color: #666; margin: 10px 0;">Doneness Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics
    st.markdown("### Detailed Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Score", f"{prediction:.4f}")
    
    with col2:
        # Calculate percentage (assuming max score is around 2.0)
        percentage = min((prediction / 2.0) * 100, 100)
        st.metric("Completion %", f"{percentage:.1f}%")
    
    with col3:
        # Temperature range
        temp_range = max(temp_lid, temp_sidewall, temp_handle, temp_innerwall) - min(temp_lid, temp_sidewall, temp_handle, temp_innerwall)
        st.metric("Temp Range", f"{temp_range:.2f}°C")
    
    with col4:
        # Average temperature
        avg_temp = (temp_lid + temp_sidewall + temp_handle + temp_innerwall) / 4
        st.metric("Avg Temp", f"{avg_temp:.2f}°C")
    
    # =========================
    # INPUT SUMMARY
    # =========================
    with st.expander("📋 Input Summary"):
        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df, width='stretch')
    
    # =========================
    # ENGINEERED FEATURES
    # =========================
    with st.expander("Engineered Features"):
        engineered_df = df_features[['Temp_Lid_Ambient_Diff', 'Temp_Handle_Ambient_Diff', 
                                      'Temp_InnerWall_Ambient_Diff', 'Temp_Range']]
        st.dataframe(engineered_df, width='stretch')

# =========================
# DISPLAY LAST PREDICTION
# =========================
if 'last_prediction' in st.session_state and not predict_button:
    st.markdown("---")
    st.markdown("## 📌 Last Prediction Summary")
    
    last_score = st.session_state['last_prediction']
    last_input = st.session_state['last_input']
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display the last prediction score prominently
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 10px; border-left: 5px solid #1f77b4;">
            <h2 style="color: #1f77b4; margin: 0;">Doneness Score</h2>
            <h1 style="color: #1f77b4; font-size: 48px; margin: 10px 0;">{last_score:.4f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Completion percentage
        percentage = min((last_score / 2.0) * 100, 100)
        st.metric("Completion", f"{percentage:.1f}%")
    
    with col2:
        # Display last input temperatures
        st.markdown("### 🌡️ Input Temperatures Used")
        
        temp_col1, temp_col2 = st.columns(2)
        
        with temp_col1:
            st.metric("Lid", f"{last_input['T1--Temp_01_Lid']:.2f}°C")
            st.metric("SideWall", f"{last_input['T1--Temp_02_SideWall']:.2f}°C")
            st.metric("Handle", f"{last_input['T1--Temp_03_Handle']:.2f}°C")
        
        with temp_col2:
            st.metric("InnerWall", f"{last_input['T1--Temp_04_InnerWall']:.2f}°C")
            st.metric("Ambient", f"{last_input['T1--Temp_05_Ambient_Temp']:.2f}°C")
    
    # Additional quick stats
    with st.expander("📊 Quick Statistics"):
        avg_temp = (last_input['T1--Temp_01_Lid'] + last_input['T1--Temp_02_SideWall'] + 
                   last_input['T1--Temp_03_Handle'] + last_input['T1--Temp_04_InnerWall']) / 4
        temp_range = max(last_input['T1--Temp_01_Lid'], last_input['T1--Temp_02_SideWall'],
                        last_input['T1--Temp_03_Handle'], last_input['T1--Temp_04_InnerWall']) - \
                    min(last_input['T1--Temp_01_Lid'], last_input['T1--Temp_02_SideWall'],
                        last_input['T1--Temp_03_Handle'], last_input['T1--Temp_04_InnerWall'])
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Average Temp", f"{avg_temp:.2f}°C")
        with stat_col2:
            st.metric("Temp Range", f"{temp_range:.2f}°C")
        with stat_col3:
            st.metric("Predicted Score", f"{last_score:.4f}")

# =========================
# QUICK EXAMPLES
# =========================
st.markdown("---")
st.markdown("## Quick Examples")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🟢 Low Doneness")
    st.code("""
Lid: 31.7°C
SideWall: 31.3°C
Handle: 30.9°C
InnerWall: 31.4°C
Ambient: 31.3°C
→ Score: ~0.25
    """)

with col2:
    st.markdown("### 🟡 Medium Doneness")
    st.code("""
Lid: 45.0°C
SideWall: 44.5°C
Handle: 43.0°C
InnerWall: 45.5°C
Ambient: 31.0°C
→ Score: ~1.0
    """)

with col3:
    st.markdown("### 🔴 High Doneness")
    st.code("""
Lid: 60.0°C
SideWall: 58.5°C
Handle: 55.0°C
InnerWall: 60.5°C
Ambient: 31.0°C
→ Score: ~1.8
    """)
