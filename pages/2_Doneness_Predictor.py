import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Doneness Predictor",
    page_icon=st.secrets['paths']['FAVICON'],
    layout="wide",
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        model         = joblib.load(st.secrets['paths']['MODEL'])
        feature_names = joblib.load(st.secrets['paths']['FEATURE_NAMES'])
        metadata      = joblib.load(st.secrets['paths']['MODEL_METADATA'])
        return model, feature_names, metadata
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, feature_names, model_metadata = load_model()

# =========================
# FEATURE ENGINEERING
# =========================
def engineer_features(water_ratio, power, cook_time, soak_time, initial_temp, rice_type):
    """Replicate the exact feature engineering from train_model.py"""
    df = pd.DataFrame([{
        'water_ratio':  water_ratio,
        'power':        power,
        'cook_time':    cook_time,
        'soak_time':    soak_time,
        'initial_temp': initial_temp,
    }])

    df['thermal_energy']   = df['power'] * df['cook_time']
    df['water_absorption'] = df['water_ratio'] * df['soak_time']
    df['temp_effect']      = (df['initial_temp'] - 20) / 15
    df['power_time']       = df['power'] * df['cook_time']

    # One-hot encode rice_type — must match training columns
    df['rice_type_basmati'] = 1 if rice_type == 'basmati' else 0
    df['rice_type_brown']   = 1 if rice_type == 'brown'   else 0
    df['rice_type_white']   = 1 if rice_type == 'white'   else 0

    return df

def doneness_label(score):
    if score < 25:
        return "Undercooked", "#E74C3C"
    elif score < 50:
        return "Lightly Cooked", "#F39C12"
    elif score < 75:
        return "Moderately Cooked", "#F1C40F"
    else:
        return "Well Cooked", "#2ECC71"

# =========================
# TITLE
# =========================
st.markdown("""
# Doneness Predictor
<div style="font-size:18px; color:gray; margin-top:-10px;">
    <b>Enter cooking parameters to predict rice doneness score (0–100)</b>
    <br><br>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ No trained model found.")
    st.info("Run `python model/train_model.py` to train the model first.")
    st.stop()

# =========================
# SIDEBAR — MODEL INFO
# =========================
# if model_metadata:
#     with st.sidebar:
#         st.markdown("### Model Info")
#         st.metric("Model Type", model_metadata.get('model_type', 'N/A'))
#         st.metric("Test R²",    f"{model_metadata.get('test_r2', 0):.4f}")
#         st.metric("MAE",        f"{model_metadata.get('mae', 0):.4f}")
#         st.metric("RMSE",       f"{model_metadata.get('rmse', 0):.4f}")

# =========================
# SESSION STATE DEFAULTS
# =========================
defaults = {
    'wr': 1.5, 'pw': 550, 'ct': 25, 'st_': 15, 'it': 27,
    'wr_n': 1.5, 'pw_n': 550, 'ct_n': 25, 'st_n': 15, 'it_n': 27,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def sync_slider(slider_key, num_key):
    st.session_state[num_key] = st.session_state[slider_key]

def sync_num(slider_key, num_key):
    st.session_state[slider_key] = st.session_state[num_key]

# =========================
# INPUT SECTION
# =========================
st.markdown("### Cooking Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    s1, n1 = st.columns([3, 1])
    with s1:
        st.slider("Water Ratio", min_value=1.0, max_value=2.0, step=0.01,
                  help="Ratio of water to rice (1.0 = equal parts, 2.0 = double water)",
                  key="wr", on_change=sync_slider, args=("wr", "wr_n"))
    with n1:
        st.number_input("Water Ratio", min_value=1.0, max_value=2.0, step=0.01,
                        key="wr_n", label_visibility="collapsed",
                        on_change=sync_num, args=("wr", "wr_n"))
    water_ratio = st.session_state["wr"]

    s2, n2 = st.columns([3, 1])
    with s2:
        st.slider("Heating Power (W)", min_value=300, max_value=799, step=1,
                  help="Heating element power in watts",
                  key="pw", on_change=sync_slider, args=("pw", "pw_n"))
    with n2:
        st.number_input("Heating Power (W)", min_value=300, max_value=799, step=1,
                        key="pw_n", label_visibility="collapsed",
                        on_change=sync_num, args=("pw", "pw_n"))
    power = st.session_state["pw"]

with col2:
    s3, n3 = st.columns([3, 1])
    with s3:
        st.slider("Cook Time (min)", min_value=10, max_value=39, step=1,
                  help="Total cooking duration in minutes",
                  key="ct", on_change=sync_slider, args=("ct", "ct_n"))
    with n3:
        st.number_input("Cook Time (min)", min_value=10, max_value=39, step=1,
                        key="ct_n", label_visibility="collapsed",
                        on_change=sync_num, args=("ct", "ct_n"))
    cook_time = st.session_state["ct"]

    s4, n4 = st.columns([3, 1])
    with s4:
        st.slider("Soak Time (min)", min_value=0, max_value=29, step=1,
                  help="Pre-soak duration before cooking",
                  key="st_", on_change=sync_slider, args=("st_", "st_n"))
    with n4:
        st.number_input("Soak Time (min)", min_value=0, max_value=29, step=1,
                        key="st_n", label_visibility="collapsed",
                        on_change=sync_num, args=("st_", "st_n"))
    soak_time = st.session_state["st_"]

with col3:
    s5, n5 = st.columns([3, 1])
    with s5:
        st.slider("Initial Temperature (°C)", min_value=20, max_value=34, step=1,
                  help="Starting temperature of water/rice",
                  key="it", on_change=sync_slider, args=("it", "it_n"))
    with n5:
        st.number_input("Initial Temperature (°C)", min_value=20, max_value=34, step=1,
                        key="it_n", label_visibility="collapsed",
                        on_change=sync_num, args=("it", "it_n"))
    initial_temp = st.session_state["it"]

    rice_type_display = st.selectbox(
        "Rice Type",
        options=['White', 'Brown', 'Basmati'],
        index=0,
        help="Type of rice being cooked"
    )
    rice_type = rice_type_display.lower()

st.markdown("---")

# =========================
# PREDICT BUTTON
# =========================
_, mid, _ = st.columns([1, 1, 1])
with mid:
    predict_button = st.button("Predict Doneness Score", type="primary", use_container_width=True)

# =========================
# PREDICTION LOGIC
# =========================
if predict_button:
    df_features = engineer_features(water_ratio, power, cook_time, soak_time, initial_temp, rice_type)
    X_input     = df_features[feature_names]
    prediction  = float(np.clip(model.predict(X_input)[0], 0, 100))

    st.session_state['last_prediction'] = prediction
    st.session_state['last_input'] = {
        'water_ratio':  water_ratio,
        'power':        power,
        'cook_time':    cook_time,
        'soak_time':    soak_time,
        'initial_temp': initial_temp,
        'rice_type':    rice_type,
    }

    st.markdown("---")
    st.markdown("### Prediction Result")

    label, color = doneness_label(prediction)

    # Score display
    r1, r2, r3 = st.columns([1, 2, 1])
    with r2:
        st.markdown(f"""
        <div style="text-align:center; padding:30px; border-radius:12px;
                    border: 2px solid {color}; margin:10px 0;">
            <h1 style="color:{color}; font-size:64px; margin:0;">{prediction:.1f}</h1>
            <p style="font-size:22px; color:gray; margin:6px 0;">out of 100</p>
            <p style="font-size:20px; font-weight:bold; color:{color};">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        number={'suffix': ' / 100', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': color},
            'steps': [
                {'range': [0,  25], 'color': '#FADBD8'},
                {'range': [25, 50], 'color': '#FDEBD0'},
                {'range': [50, 75], 'color': '#FEF9E7'},
                {'range': [75, 100],'color': '#D5F5E3'},
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'value': prediction}
        },
        title={'text': "Doneness Score"}
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    thermal = power * cook_time / 1000
    m1.metric("Doneness Score",  f"{prediction:.1f} / 100")
    m2.metric("Status",          label)
    m3.metric("Thermal Input",   f"{thermal:.2f} kJ")
    m4.metric("Water Absorption",f"{water_ratio * soak_time:.2f}")

    # Input summary
    with st.expander("Input Summary"):
        st.dataframe(pd.DataFrame([st.session_state['last_input']]), use_container_width=True)

    with st.expander("Engineered Features"):
        feat_df = df_features[['thermal_energy', 'water_absorption', 'temp_effect',
                                'power_time', 'rice_type_basmati', 'rice_type_brown', 'rice_type_white']]
        st.dataframe(feat_df, use_container_width=True)

# =========================
# LAST PREDICTION RECALL
# =========================
elif 'last_prediction' in st.session_state:
    st.markdown("---")
    last_score = st.session_state['last_prediction']
    last_input = st.session_state['last_input']
    label, color = doneness_label(last_score)

    st.markdown("### Last Prediction")
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; border-radius:10px;
                    border: 2px solid {color};">
            <h2 style="color:{color}; margin:0;">Doneness Score</h2>
            <h1 style="color:{color}; font-size:52px; margin:10px 0;">{last_score:.1f}</h1>
            <p style="color:gray;">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("#### Parameters Used")
        ic1, ic2 = st.columns(2)
        with ic1:
            st.metric("Water Ratio",  last_input['water_ratio'])
            st.metric("Power (W)",    last_input['power'])
            st.metric("Cook Time",    f"{last_input['cook_time']} min")
        with ic2:
            st.metric("Soak Time",    f"{last_input['soak_time']} min")
            st.metric("Initial Temp", f"{last_input['initial_temp']}°C")
            st.metric("Rice Type",    last_input['rice_type'].capitalize())

# =========================
# QUICK EXAMPLES
# =========================
st.markdown("---")
st.markdown("### Quick Reference")

e1, e2, e3 = st.columns(3)

with e1:
    st.markdown("#### 🔴 Undercooked")
    st.code("Water Ratio: 1.0\nPower: 300W\nCook Time: 10 min\nSoak Time: 0 min\nInitial Temp: 20°C\nRice Type: brown\n→ Doneness Score: ~0–25 (Approx.)")

with e2:
    st.markdown("#### 🟡 Moderately Cooked")
    st.code("Water Ratio: 1.5\nPower: 550W\nCook Time: 25 min\nSoak Time: 15 min\nInitial Temp: 27°C\nRice Type: white\n→ Doneness Score: ~40–65 (Approx.)")

with e3:
    st.markdown("#### 🟢 Well Cooked")
    st.code("Water Ratio: 2.0\nPower: 799W\nCook Time: 39 min\nSoak Time: 29 min\nInitial Temp: 34°C\nRice Type: white\n→ Doneness Score: ~75–100 (Approx.)")
