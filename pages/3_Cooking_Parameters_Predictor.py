import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Cooking Parameters Predictor",
    page_icon=st.secrets['paths']['FAVICON'],
    layout="wide",
)

# =========================
# LOAD OR TRAIN REVERSE MODEL
# =========================
@st.cache_resource
def load_or_train_reverse_model():
    """
    Reverse model: given a target doneness score (0-100),
    predict the cooking parameters (water_ratio, power, cook_time, soak_time, initial_temp).
    """
    reverse_model_path = st.secrets['paths']['REVERSE_MODEL']
    dataset_path       = st.secrets['paths']['REAL_TIME_DATASET']

    try:
        if os.path.exists(reverse_model_path):
            return joblib.load(reverse_model_path), "loaded"

        if not os.path.exists(dataset_path):
            return None, "no_data"

        df = pd.read_csv(dataset_path)

        X = df[['doneness']].values
        y = df[['water_ratio', 'power', 'cook_time', 'soak_time', 'initial_temp']].values

        reverse_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        reverse_model.fit(X, y)

        joblib.dump(reverse_model, reverse_model_path)
        return reverse_model, "trained"

    except Exception as e:
        st.error(f"Error building reverse model: {e}")
        return None, "error"

reverse_model, model_status = load_or_train_reverse_model()

# =========================
# TITLE
# =========================
st.markdown("""
# Cooking Parameters Predictor
<div style="font-size:18px; color:gray; margin-top:-10px;">
    <b>Enter a target doneness score to get the recommended cooking parameters</b>
    <br><br>
</div>
""", unsafe_allow_html=True)

if reverse_model is None:
    st.error("❌ Unable to load or train the reverse model.")
    if model_status == "no_data":
        st.info("Ensure `dataset/rice_doneness_dataset.csv` exists.")
    st.stop()

if model_status == "trained":
    st.success("Reverse model trained and cached for this session.")

# =========================
# SESSION STATE DEFAULTS
# =========================
if 'don' not in st.session_state:
    st.session_state['don']   = 50.0
if 'don_n' not in st.session_state:
    st.session_state['don_n'] = 50.0

def sync_don_slider():
    st.session_state['don_n'] = st.session_state['don']

def sync_don_num():
    st.session_state['don'] = st.session_state['don_n']

# =========================
# INPUT SECTION
# =========================
st.markdown("### Target Doneness Score")

_, mid, _ = st.columns([1, 2, 1])
with mid:
    sl, ni = st.columns([3, 1])
    with sl:
        st.slider(
            "Doneness Score (0–100)",
            min_value=0.0, max_value=100.0, step=0.5,
            help="Target doneness score you want to achieve",
            key="don", on_change=sync_don_slider,
        )
    with ni:
        st.number_input(
            "Doneness Score", min_value=0.0, max_value=100.0, step=0.5,
            key="don_n", label_visibility="collapsed",
            on_change=sync_don_num,
        )
    doneness_input = st.session_state["don"]

    # Doneness level indicator
    if doneness_input < 25:
        label, color = "Undercooked",        "#E74C3C"
    elif doneness_input < 50:
        label, color = "Lightly Cooked",     "#F39C12"
    elif doneness_input < 75:
        label, color = "Moderately Cooked",  "#F1C40F"
    else:
        label, color = "Well Cooked",        "#2ECC71"

    st.markdown(f"""
    <div style="text-align:center; padding:14px; border-radius:10px;
                border: 2px solid {color}; margin-top:10px;">
        <span style="font-size:20px; font-weight:bold; color:{color};">{label}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================
# PREDICT BUTTON
# =========================
_, mid2, _ = st.columns([1, 1, 1])
with mid2:
    predict_button = st.button("Predict Cooking Parameters", type="primary", use_container_width=True)

# =========================
# PREDICTION LOGIC
# =========================
if predict_button:
    X_input   = np.array([[doneness_input]])
    preds     = reverse_model.predict(X_input)[0]

    water_ratio  = float(np.clip(preds[0], 1.0, 2.0))
    power        = int(np.clip(round(preds[1]), 300, 799))
    cook_time    = int(np.clip(round(preds[2]), 10, 39))
    soak_time    = int(np.clip(round(preds[3]), 0, 29))
    initial_temp = int(np.clip(round(preds[4]), 20, 34))

    st.session_state['last_reverse'] = {
        'doneness_input': doneness_input,
        'water_ratio':    water_ratio,
        'power':          power,
        'cook_time':      cook_time,
        'soak_time':      soak_time,
        'initial_temp':   initial_temp,
        'label':          label,
        'color':          color,
    }

# =========================
# DISPLAY RESULTS
# =========================
if 'last_reverse' in st.session_state:
    r = st.session_state['last_reverse']

    st.markdown("---")
    st.markdown("### Recommended Cooking Parameters")

    # Score badge
    _, mid3, _ = st.columns([1, 2, 1])
    with mid3:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; border-radius:12px;
                    border: 2px solid {r['color']}; margin-bottom:20px;">
            <p style="color:gray; margin:0; font-size:16px;">Target Doneness Score</p>
            <h1 style="color:{r['color']}; font-size:60px; margin:4px 0;">{r['doneness_input']:.1f}</h1>
            <p style="color:{r['color']}; font-weight:bold; font-size:18px; margin:0;">{r['label']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Parameter metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Water Ratio",         f"{r['water_ratio']:.2f}")
    m2.metric("Power (W)",           f"{r['power']} W")
    m3.metric("Cook Time",           f"{r['cook_time']} min")
    m4.metric("Soak Time",           f"{r['soak_time']} min")
    m5.metric("Initial Temp",        f"{r['initial_temp']}°C")

    st.markdown("---")

    # Charts
    ch1, ch2 = st.columns(2)

    # with ch1:
    #     # Radar / bar chart of parameters (normalised 0-1)
    #     params = {
    #         'Water Ratio':    (r['water_ratio']  - 1.0) / 1.0,
    #         'Power':          (r['power']         - 300) / 499,
    #         'Cook Time':      (r['cook_time']     - 10)  / 29,
    #         'Soak Time':      r['soak_time']             / 29,
    #         'Initial Temp':   (r['initial_temp']  - 20)  / 14,
    #     }
    #     fig_bar = px.bar(
    #         x=list(params.keys()), y=list(params.values()),
    #         labels={'x': 'Parameter', 'y': 'Normalised Value (0–1)'},
    #         title='Parameter Intensity (normalised)',
    #         color=list(params.values()),
    #         color_continuous_scale='RdYlGn',
    #     )
    #     fig_bar.update_layout(coloraxis_showscale=False, showlegend=False)
    #     st.plotly_chart(fig_bar, use_container_width=True)

    # with ch2:
    #     # Gauge for doneness
    #     fig_gauge = go.Figure(go.Indicator(
    #         mode="gauge+number",
    #         value=r['doneness_input'],
    #         number={'suffix': ' / 100', 'font': {'size': 28}},
    #         gauge={
    #             'axis': {'range': [0, 100]},
    #             'bar':  {'color': r['color']},
    #             'steps': [
    #                 {'range': [0,  25], 'color': '#FADBD8'},
    #                 {'range': [25, 50], 'color': '#FDEBD0'},
    #                 {'range': [50, 75], 'color': '#FEF9E7'},
    #                 {'range': [75, 100],'color': '#D5F5E3'},
    #             ],
    #         },
    #         title={'text': "Target Doneness Score"}
    #     ))
    #     fig_gauge.update_layout(height=320, margin=dict(t=40, b=10, l=40, r=40))
    #     st.plotly_chart(fig_gauge, use_container_width=True)

    # Summary table
    with st.expander("Full Parameter Summary"):
        summary = pd.DataFrame([{
            'Target Doneness': r['doneness_input'],
            'Water Ratio':     r['water_ratio'],
            'Power (W)':       r['power'],
            'Cook Time (min)': r['cook_time'],
            'Soak Time (min)': r['soak_time'],
            'Initial Temp (°C)': r['initial_temp'],
        }])
        st.dataframe(summary, use_container_width=True)

# =========================
# QUICK EXAMPLES
# =========================
st.markdown("---")
st.markdown("### Quick Reference")

e1, e2, e3, e4 = st.columns(4)

with e1:
    st.markdown("#### 🔴 Doneness Score: 10")
    st.caption("Undercooked — low heat, short time")

with e2:
    st.markdown("#### 🟠 Doneness Score: 35")
    st.caption("Lightly cooked — moderate settings")

with e3:
    st.markdown("#### 🟡 Doneness Score: 65")
    st.caption("Moderately cooked — balanced params")

with e4:
    st.markdown("#### 🟢 Doneness Score: 90")
    st.caption("Well cooked — high power, long time")
