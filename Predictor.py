import streamlit as st

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Rice Doneness Predictor",
    page_icon=st.secrets['paths']['FAVICON'],
    layout="wide",
)

# =========================
# HERO
# =========================
st.markdown("""
# Rice Doneness Predictor
<div style="font-size:18px; color:gray; margin-top:-10px;">
    An ML-powered app to predict and optimize rice cooking
</div>
<br>
""", unsafe_allow_html=True)

st.info("ℹ️ Use the sidebar to navigate between pages.")

st.divider()

# =========================
# ABOUT
# =========================
st.markdown("### About")
st.markdown("""
This app uses a **Random Forest** model trained on rice cooking data to help you:
- Predict how well-cooked your rice will be given your cooking setup
- Reverse-engineer the ideal cooking parameters for a target doneness level
- Explore the dataset used to train the model
""")

st.divider()

# =========================
# PAGES OVERVIEW
# =========================
st.markdown("### Pages")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Dataset Explorer")
    st.markdown("Browse and filter the rice cooking training dataset. View statistics, distributions, and raw data.")

with col2:
    st.markdown("#### 🎯 Doneness Predictor")
    st.markdown("Input your cooking parameters and get a predicted doneness score (0–100) with a visual gauge.")

with col3:
    st.markdown("#### ⚙️ Cooking Parameters Predictor")
    st.markdown("Set a target doneness score and get the recommended cooking parameters to achieve it.")

st.divider()

# =========================
# MODEL INFO
# =========================
st.markdown("### How it works ?")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Input features**")
    st.markdown("""
- `water_ratio` — ratio of water to rice (1.0–2.0)
- `power` — heating element power in watts (300–799 W)
- `cook_time` — cooking duration in minutes
- `soak_time` — pre-soak duration in minutes
- `initial_temp` — starting water temperature (°C)
- `rice_type` — white, brown, or basmati
""")

with c2:
    st.markdown("**Engineered features**")
    st.markdown("""
- `thermal_energy` = power × cook_time
- `water_absorption` = water_ratio × soak_time
- `temp_effect` = (initial_temp − 20) / 15
- `power_time` = power × cook_time
""")

c3_info, _ = st.columns(2)
with c3_info:
    st.markdown("**Output**")
    st.markdown("Doneness score **0–100**, where 75–100 is well cooked.")
