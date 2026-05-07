import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Rice Doneness Predictor",
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
        
        print("[INFO] Model loaded successfully")
        return model, feature_names, metadata
    except FileNotFoundError as e:
        print(f"[INFO] No trained model found: {e}")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None, None, None

model, feature_names, model_metadata = load_model()

# =========================
# FEATURE ENGINEERING FUNCTION
# =========================
def engineer_features(df):
    """Apply the same feature engineering as training"""
    df = df.copy()
    
    # Preserve timestamp column if it exists
    has_timestamp = '_time' in df.columns
    timestamp_col = df['_time'].copy() if has_timestamp else None
    
    # Drop timestamp temporarily for feature engineering
    if '_time' in df.columns:
        df = df.drop(columns=['_time'])
    
    # Drop elapsed_minutes and Weighted_Temp if they exist
    if 'elapsed_minutes' in df.columns:
        df = df.drop(columns=['elapsed_minutes'])
    if 'Weighted_Temp' in df.columns:
        df = df.drop(columns=['Weighted_Temp'])
    
    # Create engineered features (without Weighted_Temp dependency)
    df['Temp_Lid_Ambient_Diff'] = df['T1--Temp_01_Lid'] - df['T1--Temp_05_Ambient_Temp']
    df['Temp_Handle_Ambient_Diff'] = df['T1--Temp_03_Handle'] - df['T1--Temp_05_Ambient_Temp']
    df['Temp_InnerWall_Ambient_Diff'] = df['T1--Temp_04_InnerWall'] - df['T1--Temp_05_Ambient_Temp']
    df['Temp_Range'] = df[['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 'T1--Temp_04_InnerWall']].max(axis=1) - df[['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 'T1--Temp_04_InnerWall']].min(axis=1)
    
    # Restore timestamp column if it existed
    if has_timestamp:
        df['_time'] = timestamp_col
    
    return df

# =========================
# TITLE
# =========================
st.markdown("""
# Rice Doneness Predictor
<div style="font-size:18px; color:gray; margin-top:-10px;">
    <b> Predict rice cooking doneness score </b>
    <br>
    <br>
</div>
""", unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.error("❌ No trained model found")
    st.info("Please train the model first by running: `python model/train_model.py`")
    st.stop()

# Display model info in sidebar
if model_metadata is not None:
    with st.sidebar:
        # st.markdown("## 📊 Model Information")
        # st.metric("Model Type", model_metadata['model_type'])
        # st.metric("Test R² Score", f"{model_metadata['test_r2']:.4f}")
        # st.metric("MAE", f"{model_metadata['mae']:.4f}")
        # st.metric("RMSE", f"{model_metadata['rmse']:.4f}")
        # st.metric("MAPE", f"{model_metadata['mape']:.2f}%")

        st.markdown("### 📄 Required Columns in the Dataset")
        st.markdown("""
        - T1--Temp_01_Lid
        - T1--Temp_02_SideWall
        - T1--Temp_03_Handle
        - T1--Temp_04_InnerWall
        - T1--Temp_05_Ambient_Temp
        """)

# =========================
# CSV UPLOAD SECTION
# =========================
# st.markdown("## Upload the Dataset")

uploaded_file = st.file_uploader(
    "Upload a Dataset",
    type=['csv'],
    help="Upload a CSV file containing rice cooking temperature data with required columns"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df_raw = pd.read_csv(uploaded_file)
        
        print(f"[INFO] File uploaded: {uploaded_file.name}")
        print(f"[INFO] Dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
        
        # Store in session state for persistence
        st.session_state['df_raw'] = df_raw
        st.session_state['filename'] = uploaded_file.name
        
        st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
        
        # Display dataset information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df_raw.shape[0])
        with col2:
            st.metric("Columns", df_raw.shape[1])
        with col3:
            st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        
        # Required columns for prediction
        required_cols = ['T1--Temp_01_Lid', 'T1--Temp_02_SideWall', 'T1--Temp_03_Handle', 
                       'T1--Temp_04_InnerWall', 'T1--Temp_05_Ambient_Temp']
        
        # Check if all required columns are present
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        
        if missing_cols:
            error_msg = f"Missing required columns: {', '.join(missing_cols)}"
            print(f"[INFO] {error_msg}")
            print(f"[INFO] Available columns: {', '.join(df_raw.columns.tolist())}")
            st.error(f"❌ {error_msg}")
            st.info("Please ensure your CSV contains all required temperature sensor columns.")
            
            # Show what columns are available vs required
            with st.expander("📋 Column Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Required Columns:**")
                    for col in required_cols:
                        if col in missing_cols:
                            st.markdown(f"- ❌ {col}")
                        else:
                            st.markdown(f"- ✅ {col}")
                with col2:
                    st.markdown("**Available Columns:**")
                    for col in df_raw.columns:
                        st.markdown(f"- {col}")
        else:
            print(f"[INFO] All required columns present: {', '.join(required_cols)}")
            
            # =========================
            # MAKE PREDICTIONS
            # =========================
            st.markdown("---")
            st.markdown("## Predictions")
            
            print("[INFO] Starting prediction process...")
            
            # Check for timestamp column
            if '_time' in df_raw.columns:
                print("[INFO] Timestamp column detected: _time")
            elif 'elapsed_minutes' in df_raw.columns:
                print("[INFO] Time column detected: elapsed_minutes")
            else:
                print("[INFO] No time column detected, will use sample index")
            
            with st.spinner("Generating predictions..."):
                # Engineer features
                df_features = engineer_features(df_raw)
                
                # Select only the features used in training
                X_test = df_features[feature_names]
                
                print(f"[INFO] Engineered features: {X_test.shape[1]} features for {X_test.shape[0]} samples")
                
                # Make predictions
                predictions = model.predict(X_test)
                
                print(f"[INFO] Predictions generated successfully")
                print(f"[INFO] Prediction stats - Mean: {predictions.mean():.4f}, Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
                
                # Add predictions to dataframe
                df_results = df_raw.copy()
                df_results['Predicted_Doneness_Score'] = predictions
                
                # Store results in session state
                st.session_state['df_results'] = df_results
                st.session_state['predictions'] = predictions
                
                
                # =========================
                # DISPLAY RESULTS
                # =========================
                
                # Summary statistics
                # st.markdown("### 📊 Prediction Summary")
                # col1, col2, col3, col4 = st.columns(4)
                # with col1:
                #     st.metric("Mean Doneness", f"{predictions.mean():.2f}")
                # with col2:
                #     st.metric("Min Doneness", f"{predictions.min():.2f}")
                # with col3:
                #     st.metric("Max Doneness", f"{predictions.max():.2f}")
                # with col4:
                #     st.metric("Std Dev", f"{predictions.std():.2f}")
                
                
                # Visualization tabs
                tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Distribution", "📋 Data Table"])
                
                
                with tab1:
                    st.markdown("### Doneness Score Over Time")
                    
                    # Check if timestamp is available
                    has_timestamp = '_time' in df_results.columns
                    has_elapsed = 'elapsed_minutes' in df_results.columns
                    
                    # Add toggle button if timestamp or elapsed_minutes exists
                    if has_timestamp or has_elapsed:
                        use_time = st.toggle("Use Timestamp/Time", value=True, key="time_toggle")
                        print(f"[INFO] X-axis mode: {'Timestamp/Time' if use_time else 'Sample Index'}")
                    else:
                        use_time = False
                        print("[INFO] X-axis mode: Sample Index (no time data available)")
                    
                    # Create time series plot based on toggle
                    fig = go.Figure()
                    
                    if use_time and has_timestamp:
                        # Use timestamp
                        fig.add_trace(go.Scatter(
                            x=df_results['_time'],
                            y=df_results['Predicted_Doneness_Score'],
                            mode='lines+markers',
                            name='Predicted Doneness',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=4)
                        ))
                        
                        fig.update_layout(
                            title="Rice Doneness Score Over Time",
                            xaxis_title="Timestamp",
                            yaxis_title="Doneness Score",
                            hovermode='x unified',
                            height=500
                        )
                    elif use_time and has_elapsed:
                        # Use elapsed minutes
                        fig.add_trace(go.Scatter(
                            x=df_results['elapsed_minutes'],
                            y=df_results['Predicted_Doneness_Score'],
                            mode='lines+markers',
                            name='Predicted Doneness',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=4)
                        ))
                        
                        fig.update_layout(
                            title="Rice Doneness Score Over Cooking Time",
                            xaxis_title="Elapsed Time (minutes)",
                            yaxis_title="Doneness Score",
                            hovermode='x unified',
                            height=500
                        )
                    else:
                        # Use sample index
                        fig.add_trace(go.Scatter(
                            x=df_results.index,
                            y=df_results['Predicted_Doneness_Score'],
                            mode='lines+markers',
                            name='Predicted Doneness',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=4)
                        ))
                        
                        fig.update_layout(
                            title="Rice Doneness Score Over Samples",
                            xaxis_title="Sample Index",
                            yaxis_title="Doneness Score",
                            hovermode='x unified',
                            height=500
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                
                with tab2:
                    st.markdown("### Distribution of Predicted Doneness Scores")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig_hist = px.histogram(
                            predictions, 
                            nbins=30,
                            title="Histogram of Doneness Scores",
                            labels={'value': 'Doneness Score', 'count': 'Frequency'}
                        )
                        fig_hist.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig_box = px.box(
                            y=predictions,
                            title="Box Plot of Doneness Scores",
                            labels={'y': 'Doneness Score'}
                        )
                        fig_box.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_box, use_container_width=True)
                
                
                with tab3:
                    st.markdown("### Predictions")
                    
                    # Display options
                    show_all_cols = st.checkbox("Show all columns", value=False)
                    
                    if show_all_cols:
                        display_df = df_results
                    else:
                        # Show only key columns
                        key_cols = ['Predicted_Doneness_Score']
                        if 'elapsed_minutes' in df_results.columns:
                            key_cols.insert(0, 'elapsed_minutes')
                        if 'Weighted_Temp' in df_results.columns:
                            key_cols.insert(1, 'Weighted_Temp')
                        available_cols = [col for col in key_cols if col in df_results.columns]
                        display_df = df_results[available_cols]
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{uploaded_file.name}",
                        mime="text/csv"
                    )
        
        # Display dataset preview
        with st.expander("🔍 View Raw Data Preview"):
            st.dataframe(df_raw.head(20), use_container_width=True)
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"[INFO] {error_msg}")
        st.error(f"❌ {error_msg}")
        st.info("Please ensure your file is a valid CSV format with the required columns.")
else:
    st.info("Upload a CSV file to get started")
    
    # Show example format
    with st.expander("📋 Sample Dataset Format (.csv)"):
        example_data = {
            'T1--Temp_01_Lid': [31.7, 31.7, 31.7],
            'T1--Temp_02_SideWall': [31.3, 31.3, 31.3],
            'T1--Temp_03_Handle': [30.9, 30.8, 30.8],
            'T1--Temp_04_InnerWall': [31.4, 31.4, 31.4],
            'T1--Temp_05_Ambient_Temp': [31.3, 31.3, 31.3]
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)