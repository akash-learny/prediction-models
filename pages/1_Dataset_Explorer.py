import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="",
    layout="wide"
)


# =========================
# TITLE
# =========================
st.markdown("""
# Dataset Browser
<div style="font-size:18px; color:gray; margin-top:-10px;">
    <b> View real-time rice cooker temperature data</b> <br>
    <b> NOTE: This is a Dataset used for Training the Model </b>
    <br>
    <br>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_dataset():
    """Load the real-time dataset from .env path"""
    try:
        dataset_path = os.getenv('REAL_TIME_DATASET', 'dataset/Electric rice cooker_temperature rise test.csv')
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"[INFO] Dataset not found at: {dataset_path}")
            return None, f"Dataset not found at: {dataset_path}"
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"[INFO] Dataset loaded successfully: {dataset_path}")
        print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df, None
    except Exception as e:
        error_msg = f"Error loading dataset: {str(e)}"
        print(f"[INFO] {error_msg}")
        return None, error_msg

# Load the dataset
df, error = load_dataset()

# =========================
# DISPLAY DATASET
# =========================
if error:
    st.error(f"❌ {error}")
    st.info("Please ensure the dataset file exists at the path specified in .env file")
    st.code(f"REAL_TIME_DATASET={os.getenv('REAL_TIME_DATASET', 'dataset/Electric rice cooker_temperature rise test.csv')}")
else:
    # Dataset information
    st.markdown("### 📈 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("File Size", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Display dataset path
    # st.info(f"📁 Dataset Path: `{os.getenv('REAL_TIME_DATASET')}`")
    
    st.markdown("---")
    
    # =========================
    # FILTERS AND OPTIONS
    # =========================
    st.markdown("### 🔧 Display Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Number of rows to display
        num_rows = st.selectbox(
            "Rows to display",
            options=["All", "10", "50", "100", "500", "1000"],
            index=0
        )
    
    with col2:
        # Column selection
        show_all_cols = st.checkbox("Show all columns", value=True)
    
    with col3:
        # Search functionality
        search_enabled = st.checkbox("Enable search", value=False)
    
    # Column filter
    if not show_all_cols:
        selected_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:5]
        )
    else:
        selected_columns = df.columns.tolist()
    
    # Search filter
    if search_enabled:
        search_term = st.text_input("Search in dataset (searches all columns)")
        if search_term:
            # Search across all columns
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            filtered_df = df[mask]
            st.info(f"Found {len(filtered_df)} rows matching '{search_term}'")
        else:
            filtered_df = df
    else:
        filtered_df = df
    
    st.markdown("---")
    
    # =========================
    # DISPLAY TABLE
    # =========================
    st.markdown("### 📋 Dataset Table")
    
    # Apply row limit
    if num_rows == "All":
        display_df = filtered_df[selected_columns]
    else:
        display_df = filtered_df[selected_columns].head(int(num_rows))
    
    # Display dataframe
    st.dataframe(
        display_df,
        width='stretch',
        height=600
    )
    
    # Display row count
    st.caption(f"Showing {len(display_df)} of {len(filtered_df)} rows")
    
    st.markdown("---")
    
    # =========================
    # COLUMN INFORMATION
    # =========================
    with st.expander("📊 Column Information"):
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str).values,  # Convert to string to avoid Arrow serialization issues
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, width='stretch')
    
    # =========================
    # STATISTICAL SUMMARY
    # =========================
    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe(), width='stretch')
    
    # =========================
    # DOWNLOAD OPTIONS
    # =========================
    # st.markdown("---")
    # st.markdown("### 💾 Download Options")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     # Download filtered data
    #     csv = display_df.to_csv(index=False)
    #     st.download_button(
    #         label="📥 Download Displayed Data as CSV",
    #         data=csv,
    #         file_name="filtered_dataset.csv",
    #         mime="text/csv"
    #     )
    
    # with col2:
    #     # Download full dataset
    #     csv_full = df.to_csv(index=False)
    #     st.download_button(
    #         label="📥 Download Full Dataset as CSV",
    #         data=csv_full,
    #         file_name="full_dataset.csv",
    #         mime="text/csv"
    #     )

# # =========================
# # SIDEBAR INFO
# # =========================
# with st.sidebar:
#     st.markdown("")
#     st.markdown("---")
#     st.markdown("")
