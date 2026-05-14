import streamlit as st
import pandas as pd
import plotly.express as px
import os

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Dataset Explorer",
    page_icon=st.secrets['paths']['FAVICON'],
    layout="wide"
)

# =========================
# TITLE
# =========================
st.markdown("""
# Dataset Browser
<div style="font-size:18px; color:gray; margin-top:-10px;">
    <b>View and explore the rice doneness training dataset</b>
    <br><br>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_dataset():
    dataset_path = st.secrets['paths']['REAL_TIME_DATASET']
    try:
        if not os.path.exists(dataset_path):
            return None, f"Dataset not found at: {dataset_path}"
        df = pd.read_csv(dataset_path)
        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_dataset()

# =========================
# ERROR STATE
# =========================
if error:
    st.error(f"❌ {error}")
    st.stop()

# =========================
# METRICS
# =========================
st.markdown("### Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows",    f"{df.shape[0]:,}")
c2.metric("Total Columns", df.shape[1])
c3.metric("Rice Types",    df['rice_type'].nunique())
# c4.metric("Avg Doneness",  f"{df['doneness'].mean():.1f}")
c4.metric("Missing Values", df.isnull().sum().sum())

st.markdown("---")

# =========================
# FILTERS
# =========================
st.markdown("### Filters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    rice_options = ["All"] + sorted(df['rice_type'].unique().tolist())
    rice_filter = st.selectbox("Rice Type", options=rice_options)

with col2:
    doneness_range = st.slider(
        "Doneness Range",
        min_value=float(df['doneness'].min()),
        max_value=float(df['doneness'].max()),
        value=(float(df['doneness'].min()), float(df['doneness'].max())),
        step=0.1
    )

with col3:
    power_range = st.slider(
        "Power (W)",
        min_value=int(df['power'].min()),
        max_value=int(df['power'].max()),
        value=(int(df['power'].min()), int(df['power'].max()))
    )

with col4:
    num_rows = st.selectbox("Rows to display", ["All", "10", "50", "100", "500"], index=0)

# Apply filters
filtered_df = df.copy()
if rice_filter != "All":
    filtered_df = filtered_df[filtered_df['rice_type'] == rice_filter]
filtered_df = filtered_df[
    (filtered_df['doneness'] >= doneness_range[0]) &
    (filtered_df['doneness'] <= doneness_range[1]) &
    (filtered_df['power'] >= power_range[0]) &
    (filtered_df['power'] <= power_range[1])
]

st.caption(f"Showing {len(filtered_df):,} of {len(df):,} rows after filters")

st.markdown("---")

# =========================
# CHARTS
# =========================
# st.markdown("### Distributions")

# ch1, ch2 = st.columns(2)

# with ch1:
#     fig = px.histogram(
#         filtered_df, x='doneness', color='rice_type',
#         nbins=40, barmode='overlay', opacity=0.7,
#         title='Doneness Score Distribution by Rice Type',
#         labels={'doneness': 'Doneness Score (0–100)', 'rice_type': 'Rice Type'},
#         color_discrete_map={'white': '#4C9BE8', 'brown': '#C8813A', 'basmati': '#6DBF67'}
#     )
#     fig.update_layout(legend_title_text='Rice Type')
#     st.plotly_chart(fig, use_container_width=True)

# with ch2:
#     fig2 = px.scatter(
#         filtered_df, x='cook_time', y='doneness', color='rice_type',
#         opacity=0.6, title='Cook Time vs Doneness',
#         labels={'cook_time': 'Cook Time (min)', 'doneness': 'Doneness Score', 'rice_type': 'Rice Type'},
#         color_discrete_map={'white': '#4C9BE8', 'brown': '#C8813A', 'basmati': '#6DBF67'}
#     )
#     st.plotly_chart(fig2, use_container_width=True)

# ch3, ch4 = st.columns(2)

# with ch3:
#     fig3 = px.scatter(
#         filtered_df, x='water_ratio', y='doneness', color='rice_type',
#         opacity=0.6, title='Water Ratio vs Doneness',
#         labels={'water_ratio': 'Water Ratio', 'doneness': 'Doneness Score', 'rice_type': 'Rice Type'},
#         color_discrete_map={'white': '#4C9BE8', 'brown': '#C8813A', 'basmati': '#6DBF67'}
#     )
#     st.plotly_chart(fig3, use_container_width=True)

# with ch4:
#     avg_by_rice = filtered_df.groupby('rice_type')['doneness'].mean().reset_index()
#     fig4 = px.bar(
#         avg_by_rice, x='rice_type', y='doneness',
#         title='Average Doneness by Rice Type',
#         labels={'rice_type': 'Rice Type', 'doneness': 'Avg Doneness Score'},
#         color='rice_type',
#         color_discrete_map={'white': '#4C9BE8', 'brown': '#C8813A', 'basmati': '#6DBF67'}
#     )
#     fig4.update_layout(showlegend=False)
#     st.plotly_chart(fig4, use_container_width=True)

# st.markdown("---")

# =========================
# DATA TABLE
# =========================
st.markdown("### Dataset Table")

display_df = filtered_df if num_rows == "All" else filtered_df.head(int(num_rows))
st.dataframe(display_df, use_container_width=True, height=500)
st.caption(f"Showing {len(display_df):,} of {len(filtered_df):,} filtered rows")

st.markdown("---")

# =========================
# COLUMN INFO & STATS
# =========================
with st.expander("Column Information"):
    col_info = pd.DataFrame({
        'Column':        df.columns,
        'Type':          df.dtypes.astype(str).values,
        'Non-Null':      df.count().values,
        'Null':          df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

with st.expander("Statistical Summary"):
    st.dataframe(df.describe(), use_container_width=True)
