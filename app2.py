# -*- coding: utf-8 -*-
"""
Mortality Risk Predictor for Acute Ischemic Stroke with SHAP Explanations
==============================================

This Streamlit application predicts patient mortality risk based on 8 clinical features
and provides SHAP (SHapley Additive exPlanations) visualizations to explain how each feature
influences the prediction.

Key Features:
- Interactive input of 8 clinical parameters
- Real-time mortality probability prediction
- Color-coded risk assessment (low/medium/high)
- SHAP force plot for intuitive feature impact visualization
- Waterfall plot showing feature contribution hierarchy
- Robust error handling and input validation
- Performance optimizations with caching

Features Included:
- D-dimer: Thrombosis marker (μg/mL)
- AG (Anion Gap): Acid-base balance indicator (mmol/L)
- Age: Patient age in years
- NIHSS: Neurological deficit severity score
- Time: Time from onset to admission (hours)
- WBC: White Blood Cell count (×10⁹/L)
- NSE: Neuron-Specific Enolase (ng/mL) - Brain injury marker
- Glucose: Blood Glucose level (mmol/L)

Dependencies:
- streamlit: Web application framework
- pandas: Data manipulation
- numpy: Numerical calculations
- shap: Machine learning model explainability
- matplotlib: Plotting and visualization
- joblib: Model serialization/deserialization

Required Files:
- death_model.pkl: Trained XGBoost classifier for mortality prediction
- scaler_model.pkl: Feature scaler for input normalization (optional)

Usage:
1. Run the app: streamlit run app2.py
2. Input patient clinical features in the sidebar
3. View predicted mortality probability and risk level
4. Explore SHAP visualizations to understand feature contributions

Note: This application is for educational and research purposes only. Not intended for clinical decision-making.
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import streamlit.components.v1 as components

# Configuration and Setup
st.set_page_config(
    page_title="Mortality Risk Predictor for Acute Ischemic Stroke",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.cache_resource.clear()
plt.rcParams['font.family'] = 'SimHei'

# App title and description with Q&A format
st.title("🩺 Mortality Risk Predictor for Acute Ischemic Stroke")
st.markdown("---")

# ==========================
# Feature Definitions
# ==========================

# List of all clinical features used in the model
FEATURES = ['D-dimer', 'AG', 'Age', 'NIHSS', 'Time', 'WBC', 'NSE', 'Glucose']

# Descriptions of each feature with units and clinical significance
FEATURE_DESCRIPTIONS = {
    'D-dimer': 'D-dimer, protein fragment produced by blood clot breakdown, marker of thrombosis (Range: 0.0-20.0 mg/L)',
    'AG': 'AG, indicator of liver function, nutrition, and inflammation (Range: 0.0-25.0 %)',
    'Age': 'Patient age in years (Range: 0-100 years)',
    'NIHSS': 'NIHSS score, scale assessed for ischemic stroke severity (Range: 0-42)',
    'Time': 'assessment of pre-hospital delay and thrombolysis time window (Range: 0-168 hours)',
    'WBC': 'WBC, non-specific marker of systemic infection and inflammation (Range: 0.0-30.0 ×10^9/L)',
    'NSE': 'NSE, marker of neuronal damage and neurologic deficit (Range: 0.0-100.0 ng/mL)',
    'Glucose': 'Fasting blood glucose, marker of glucose metabolism (Range: 0.0-30.0 mmol/L)'
}

# Configuration for input widgets (type, range, default value, step, formatting)
FEATURE_CONFIGS = {
    'D-dimer':  dict(kind='float', min=0.0,  max=20.0, value=0.0,   step=0.1, fmt="%.2f"),  # Thrombosis marker
    'AG':       dict(kind='float', min=0.0,  max=25.0, value=0.0,   step=0.1, fmt="%.2f"),  # Acid-base balance
    'Age':      dict(kind='int',   min=0,    max=100,  value=0,     step=1),              # Age in years
    'NIHSS':    dict(kind='int',   min=0,    max=42,   value=0,     step=1),              # Neurological deficit
    'Time':     dict(kind='int',   min=0,    max=168,  value=0,     step=1),              # Onset to admission
    'WBC':      dict(kind='float', min=0.0,  max=30.0, value=0.0,   step=0.1, fmt="%.2f"),  # Immune response
    'NSE':      dict(kind='float', min=0.0,  max=100.0,value=0.0,   step=0.1, fmt="%.2f"),  # Brain injury
    'Glucose':  dict(kind='float', min=0.0,  max=30.0, value=0.0,   step=0.1, fmt="%.2f")   # Metabolic marker
}

# Set of features that should NOT be scaled (categorical or ordinal features)
# NIHSS is kept as raw value because it has clinical interpretability as a scale score
NON_SCALED_FEATURES = {"NIHSS"}



@st.cache_resource
def load_model_and_scaler():
    """
    Load the pre-trained model and scaler from disk.
    
    Returns:
        tuple: (model, scaler) where model is the XGBoost classifier and scaler is the feature scaler
    """
    # Try multiple possible paths for model files to improve portability
    model_paths = [
        './death_model.pkl'
    ]
    
    scaler_paths = [
        './scaler_model.pkl'
    ]
    
    # Load model
    mdl = None
    for path in model_paths:
        try:
            mdl = joblib.load(path)["model"]
            if mdl:
                break
        except Exception as e:
            st.warning(f"Failed to load model from {path}: {e}")
            continue
    
    if not mdl:
        st.error("Could not load model from any of the specified paths. Please check the model file.")
        return None, None
    
    # Load scaler
    scl = None
    for path in scaler_paths:
        try:
            scl = joblib.load(path)
            if scl:
                break
        except Exception as e:
            st.warning(f"Failed to load scaler from {path}: {e}")
            continue
    
    if not scl:
        st.info("No scaler found. Using features without scaling.")
    
    return mdl, scl


@st.cache_data(show_spinner=False)
def preprocess_features(sample_df, _model, _scaler, non_scaled_features=NON_SCALED_FEATURES):
    """
    Preprocess input features by scaling and ensuring proper format for model prediction.
    Uses caching to avoid redundant processing for identical inputs.
    
    Args:
        sample_df: DataFrame containing input features
        _model: The trained machine learning model (leading underscore prevents hashing)
        _scaler: The feature scaler (if available, leading underscore prevents hashing)
        non_scaled_features: Set of features that should not be scaled
        
    Returns:
        DataFrame: Preprocessed features ready for model prediction
    """
    try:
        # Convert sample_df to a tuple of values for caching key purposes
        sample_tuple = tuple(sample_df.values.flatten())
        
        # Data validation: Ensure all input values are finite
        if not np.isfinite(sample_tuple).all():
            st.error("Input values contain invalid numbers (NaN or infinity). Please check your inputs.")
            st.stop()
        
        # 1) Get the complete list of features used during model training
        # This ensures consistent feature set and order
        expected_features = None
        
        # Try multiple approaches to get feature names based on model type
        try:
            if hasattr(_model, 'get_booster'):
                expected_features = _model.get_booster().feature_names
            elif hasattr(_model, 'feature_names_in_'):
                expected_features = _model.feature_names_in_
            elif hasattr(_model, 'coef_') and len(_model.coef_) > 0:
                # Linear models may have coef_ but no feature names
                expected_features = [f'feature_{i}' for i in range(len(_model.coef_))]
        except Exception as e:
            st.warning(f"Could not retrieve feature names from model: {e}")
        
        if not expected_features or len(expected_features) == 0:
            # Fallback: Use current input columns if model features can't be retrieved
            expected_features = list(sample_df.columns)
            st.warning(f"Using input feature names as fallback: {expected_features}")

        # 2) Prepare raw input values for all expected features
        # Missing features are filled with scaler mean (if available) or 0.0
        scaler_means = {}
        if _scaler is not None:
            if hasattr(_scaler, "feature_names_in_") and hasattr(_scaler, "mean_"):
                scaler_means = {name: m for name, m in zip(_scaler.feature_names_in_, _scaler.mean_)}
            elif hasattr(_scaler, "mean_"):  # Some scalers don't have feature_names_in_
                scaler_means = {expected_features[i]: _scaler.mean_[i] 
                              for i in range(min(len(_scaler.mean_), len(expected_features)))}        
        
        raw_full = {}
        for col in expected_features:
            if col in sample_df.columns:
                raw_full[col] = float(sample_df[col].iloc[0])
            else:
                default_val = scaler_means.get(col, 0.0)
                raw_full[col] = float(default_val)
                st.info(f"Feature '{col}' not provided, using default value: {default_val}")

        # 3) Only scale non-categorical features
        # First transform according to scaler's column order, then map back to model's expected features
        scaled_map = {}
        if _scaler is not None:
            try:
                if hasattr(_scaler, "feature_names_in_"):
                    # Create a row of raw data in scaler's feature order
                    scaler_row = []
                    for col in _scaler.feature_names_in_:
                        if col in raw_full:
                            scaler_row.append(float(raw_full[col]))
                        else:
                            # Use scaler mean for features not in input but required by scaler
                            scaler_row.append(float(scaler_means.get(col, 0.0)))
                    
                    scaler_row = np.array([scaler_row])
                    scaled = _scaler.transform(scaler_row)
                    scaled_map = {col: float(scaled[0, i]) for i, col in enumerate(_scaler.feature_names_in_)}
                    
                    # Remove scaled values for non-scaled features
                    for feature in non_scaled_features:
                        if feature in scaled_map:
                            scaled_map.pop(feature, None)
                else:
                    st.warning("Scaler doesn't have feature names, skipping scaling.")
            except Exception as e:
                st.warning(f"Feature scaling failed: {e}. Using unscaled features.")

        # 4) Assemble the final DataFrame for model prediction
        # Use original values for non-scaled features, scaled values otherwise
        final_row = []
        for col in expected_features:
            if col in non_scaled_features:
                val = float(raw_full.get(col, 0.0))              # Non-scaled feature: use original value
            elif col in scaled_map:
                val = float(scaled_map[col])                     # Scaled feature: use scaled value
            else:
                val = float(raw_full.get(col, 0.0))              # Fallback: use original/mean value
            final_row.append(val)

        processed_df = pd.DataFrame([final_row], columns=expected_features)
        
        # Verify processed data is valid
        if not np.isfinite(processed_df.values).all():
            st.error("Processed features contain invalid numbers. Please check your inputs.")
            st.stop()
            
        return processed_df
    except Exception as e:
        st.error(f"Failed to process features: {e}")
        st.stop()



@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    """
    Create and cache a SHAP TreeExplainer for the model to avoid repeated initialization.
    
    Args:
        _model: The trained machine learning model (underscore to prevent caching)
        
    Returns:
        shap.TreeExplainer: Cached SHAP explainer object
    """
    return shap.TreeExplainer(_model)

@st.cache_data(show_spinner=False)
def generate_shap_explanations(_model, X_model_df, sample_df, features=FEATURES):
    """
    Generate SHAP explanations for the model prediction.
    Uses caching to avoid redundant calculations for identical inputs.
    
    Args:
        _model: The trained machine learning model (underscore to prevent caching)
        X_model_df: Preprocessed features for prediction
        sample_df: Original input features
        features: List of feature names
        
    Returns:
        tuple: (base_value, shap_used) where base_value is the expected value and shap_used are the SHAP values
    """
    try:
        # Get cached SHAP explainer
        explainer = get_shap_explainer(_model)
        
        # Calculate SHAP values
        shap_vals = explainer.shap_values(X_model_df)
        
        # Handle both binary and multi-class cases
        if isinstance(shap_vals, list):
            shap_used = shap_vals[1]  # Use positive class (mortality) for binary classification
        else:
            shap_used = shap_vals
        
        # Get base value (expected value)
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_value = float(exp_val[1])
        else:
            base_value = float(exp_val)
        
        return base_value, shap_used
    except Exception as e:
        st.error(f"Failed to generate SHAP explanations: {e}")
        st.stop()

def st_shap(plot, height=None):
    """
    Display a SHAP plot in Streamlit.
    
    Args:
        plot: The SHAP plot object to display
        height: Optional height for the plot
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Initialize session state for prediction trigger and reset flag
if 'predict' not in st.session_state:
    st.session_state.predict = False
if 'reset' not in st.session_state:
    st.session_state.reset = False

# Add page state management
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'input'  # Possible values: 'input', 'results'

# Note: Removed explicit session state initialization for input fields to fix Streamlit warning
# Streamlit automatically creates session state entries for widgets with key parameters

# Load model and scaler at the beginning to ensure it's available for both pages
model, scaler = load_model_and_scaler()
if model is None:
    st.stop()

# Page routing logic
if st.session_state.current_page == 'input':
    # Q&A section - only show on input page
    st.markdown("""
    **Q: What is this app?**

    A: This app offers a practical tool for assessing the prognosis of acute ischemic stroke. It is designed for patients aged ≥18 years who had ischemic stroke within 7 days of symptom onset.

    **Q: What does it do?**

    A: With the 8 features you provide, we will quickly estimate <font color='red'>**your risk of mortality after ischemic stroke**</font> and use clear charts to show how each feature affects the result.

    **Q: How do I use it?**

    A: Fill in the patient's clinical features and click the predict button to get the result.

    **Q: Can I trust the number?**

    A: The model was trained on thousands of real records, but bodies vary. <font color='red'>**It's a reference, not a verdict**</font>—bring it to your doctor for discussion.
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Main Page: Input controls
    st.header("Input Clinical Features")

    # Add a reset button using session state
    if st.button('Reset Inputs'):
        # Reset all input fields to their default values (0)
        for feature in FEATURE_CONFIGS:
            st.session_state[f"{feature}_input"] = FEATURE_CONFIGS[feature]['value']
        st.session_state.predict = False
        st.rerun()

    # Create two columns in the main page for better organization
    col1, col2 = st.columns([5, 5], gap="large")

    input_values = {}

    # Group features into logical categories
    laboratory_features = ['D-dimer', 'AG', 'WBC', 'NSE', 'Glucose']
    clinical_features = ['Age', 'NIHSS', 'Time']

    # Add section headers
    for feature in laboratory_features:
        cfg = FEATURE_CONFIGS[feature]
        
        # Define labels with specific units as required
        if feature == 'D-dimer':
            label = "D-dimer, mg/L"
        elif feature == 'AG':
            label = "The albumin to globulin ratio (AG), %"
        elif feature == 'WBC':
            label = "White blood cell count (WBC), 10^9/L"
        elif feature == 'NSE':
            label = "Neuron-specific enolase (NSE), ng/mL"
        elif feature == 'Glucose':
            label = "Fasting blood glucose, mmol/L"
        else:
            label = feature
        
        # Determine which column to place the feature in
        with (col1 if feature in ['D-dimer', 'WBC', 'Glucose'] else col2):
            if cfg['kind'] == 'float':
                input_values[feature] = st.number_input(
                    label=label,
                    min_value=float(cfg['min']),
                    max_value=float(cfg['max']),
                    # Removed value parameter to fix Streamlit warning
                    step=float(cfg['step']),
                    format=cfg.get('fmt', "%.2f"),
                    help=FEATURE_DESCRIPTIONS[feature],
                    key=f"{feature}_input"
                )
            else:
                input_values[feature] = st.number_input(
                    label=label,
                    min_value=int(cfg['min']),
                    max_value=int(cfg['max']),
                    # Removed value parameter to fix Streamlit warning
                    step=int(cfg['step']),
                    help=FEATURE_DESCRIPTIONS[feature],
                    key=f"{feature}_input"
                )

    for feature in clinical_features:
        cfg = FEATURE_CONFIGS[feature]
        
        # Define labels with specific units as required
        if feature == 'Age':
            label = "Age, years"
        elif feature == 'NIHSS':
            label = "NIHSS score at baseline"
        elif feature == 'Time':
            label = "Time from onset to hospitalization, hours"
        else:
            label = feature
        
        with (col1 if feature in ['Age'] else col2):
            if cfg['kind'] == 'float':
                input_values[feature] = st.number_input(
                    label=label,
                    min_value=float(cfg['min']),
                    max_value=float(cfg['max']),
                    # Removed value parameter to fix Streamlit warning
                    step=float(cfg['step']),
                    format=cfg.get('fmt', "%.2f"),
                    help=FEATURE_DESCRIPTIONS[feature],
                    key=f"{feature}_input"
                )
            else:
                input_values[feature] = st.number_input(
                    label=label,
                    min_value=int(cfg['min']),
                    max_value=int(cfg['max']),
                    # Removed value parameter to fix Streamlit warning
                    step=int(cfg['step']),
                    help=FEATURE_DESCRIPTIONS[feature],
                    key=f"{feature}_input"
                )

    # Add predict button in main page
    st.markdown("---")
    if st.button("Predict", type="primary", use_container_width=True):
        # Validate input values before proceeding to results page
        validation_error = False
        for feature, value in input_values.items():
            cfg = FEATURE_CONFIGS[feature]
            if cfg['kind'] == 'float':
                min_val = float(cfg['min'])
                max_val = float(cfg['max'])
            else:
                min_val = int(cfg['min'])
                max_val = int(cfg['max'])
            
            if value < min_val:
                st.error(f"{feature} value cannot be less than {min_val}, please re-enter!")
                validation_error = True
            elif value > max_val:
                st.error(f"{feature} value cannot be greater than {max_val}, please re-enter!")
                validation_error = True
        
        if not validation_error:
            # Create DataFrame from input values and store in session state
            sample_df = pd.DataFrame([input_values], columns=FEATURES)
            st.session_state.sample_df = sample_df
            st.session_state.predict = True
            st.session_state.current_page = 'results'
            st.rerun()

elif st.session_state.current_page == 'results':
    # Only show prediction results if predict button was clicked and we're on results page
    if st.session_state.predict:
        # Get the sample_df from session state
        if 'sample_df' in st.session_state:
            sample_df = st.session_state.sample_df
        else:
            st.error("No input data found. Please return to the input page and try again.")
            st.stop()
        
        # Use a spinner to show processing status
        with st.spinner('Processing input features and generating prediction...'):
            # Preprocess features for prediction
            X_model_df = preprocess_features(sample_df, model, scaler)
            
            # Generate SHAP explanations
            base_value, shap_used = generate_shap_explanations(model, X_model_df, sample_df)

        # Make prediction and display results
        st.subheader("Prediction Result")
        
        try:
            # Validate model has predict_proba method
            if not hasattr(model, 'predict_proba'):
                # Fallback for models without predict_proba (e.g., some SVMs)
                prediction = model.predict(X_model_df)[0]
                mortality_prob = 100.0 if prediction == 1 else 0.0
                st.warning("Model doesn't support probability prediction. Showing binary result.")
            else:
                proba = model.predict_proba(X_model_df)[0]
                if len(proba) < 2:
                    st.error("Model prediction output format is unexpected. Expected at least 2 classes.")
                    st.stop()
                mortality_prob = float(proba[1]) * 100.0
            
            # Ensure probability is within valid range
            mortality_prob = max(0.0, min(100.0, mortality_prob))
            
            # Display results according to new layout requirements with color-coded probability
            color = "red" if mortality_prob >= 50 else "green"
            st.markdown(f"<div style='font-size: 1.2em;'><strong>Based on feature values, your predicted possiblity of mortality after ischemic stroke is <font color='{color}'>{mortality_prob:.2f}%</font>.</strong></div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # ==========================   
        # Results Summary Section      
        # ==========================   

        # Explain what SHAP values represent with clear bullet points
        #st.markdown("**SHAP values show how each feature influences the prediction:**")
        #st.markdown("- Red bars/areas: Features that increase mortality risk")
        #st.markdown("- Blue bars/areas: Features that decrease mortality risk")
        #st.markdown("- Length/intensity: Magnitude of feature's influence")

        # Display SHAP force plot - shows feature contributions for this specific prediction
        try:
            # Validate that SHAP values match the number of features
            if shap_used.shape[1] != len(FEATURES):
                st.warning(f"SHAP values shape ({shap_used.shape[1]}) doesn't match feature count ({len(FEATURES)}). Using available features.")
                

            # Create columns to limit the width of the waterfall plot
            col1, col2 = st.columns([0.5, 0.5])
            
            with col1:
                # Fix font display issues for negative values
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['axes.unicode_minus'] = False
                
                # Create a new figure with smaller size and constrained layout to avoid warnings
                fig, ax = plt.subplots(figsize=(4, 2.5), constrained_layout=True)
                
                try:
                    # Create SHAP Explanation object for waterfall plot
                    # This object contains all necessary data for visualization
                    shap_explanation = shap.Explanation(
                        values=shap_used[0, :],               # SHAP values
                        base_values=base_value,               # Base (expected) value
                        data=sample_df.iloc[0, :].values,     # Original feature values
                        feature_names=FEATURES                # Feature names
                    )
                    
                    # Generate waterfall plot showing top feature contributions
                    # max_display: Shows up to 8 most influential features for better fit
                    shap.plots.waterfall(
                        shap_explanation,
                        show=False,
                        max_display=8
                    )
                    
                    # Display the plot in Streamlit
                    st.pyplot(fig)
                except Exception as inner_e:
                    # Handle errors specific to waterfall plot generation
                    st.error(f"Waterfall plot failed: {inner_e}")
                finally:
                    # Explicitly close the figure to free memory
                    plt.close(fig)
            
        # Handle general visualization
        except Exception as e:
            st.error(f"Failed to display SHAP visualizations: {e}")
            # Provide helpful information for troubleshooting
            st.info("For detailed model explanation, ensure SHAP is properly installed and compatible with your model.")
        
        # Add risk assessment message based on probability after the plot
        if mortality_prob >= 50:
            st.markdown("<div style='font-size: 1.2em;'>You may have a <font color='red'><strong>high risk</strong></font> of mortality after ischemic stroke—seek medical help immediately and ensure close monitoring!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 1.2em;'>Congratulations—your risk of mortality is <font color='green'><strong>low</strong></font>. Please continue to follow your doctor's rehabilitation plan closely!</div>", unsafe_allow_html=True)
        
        # Add a back button to return to input page
        st.markdown("---")
        if st.button("Back to Input Page", use_container_width=True):
            st.session_state.current_page = 'input'
            st.session_state.predict = False
            # Keep the input values in session state so they're preserved when returning
            st.rerun()

# Add a "How to Use" section to sidebar
st.sidebar.markdown("---")
st.sidebar.header("How to Use")
st.sidebar.markdown("1. Enter patient's clinical features in the input fields")
st.sidebar.markdown("2. Click 'Reset Inputs' to clear all fields")
st.sidebar.markdown("3. View the mortality probability prediction")
st.sidebar.markdown("4. Interpret results using SHAP visualizations")
st.sidebar.markdown("5. Features in red increase risk, blue decrease risk")

# ==========================
# Sidebar Information Sections
# ==========================

# Add model information to sidebar
with st.sidebar.expander("📋 Model Information"):
    st.markdown(f"**Model Type:** XGBoost Classifier")
    st.markdown(f"**Number of Features:** {len(FEATURES)}")
    try:
        st.markdown(f"**Base Value:** {base_value:.4f}")
    except:
        st.markdown("**Base Value:** Not available")
    st.markdown("- Base value represents model's average prediction")
    st.markdown("- Features shift prediction from this baseline")


# ==========================
# Main Function Entry Point
# ==========================

# Ensure the app runs only when executed directly
if __name__ == "__main__":
    # This is the main entry point for the Streamlit app
    # All the code above will be executed when running: streamlit run app2.py
    pass

