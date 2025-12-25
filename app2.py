# -*- coding: utf-8 -*-
"""
Mortality Risk Predictor with SHAP Explanations
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
    page_title="Mortality Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.cache_resource.clear()
plt.rcParams['font.family'] = 'SimHei'

# App title and description with improved formatting
st.title("🩺 Mortality Risk Predictor")
st.markdown("---")

# Add a brief introduction
st.markdown("""
This application predicts patient mortality risk based on clinical features and provides **SHAP (SHapley Additive exPlanations)**
visualizations to explain how each feature influences the prediction. 

**Features include:**
- Interactive input of 8 clinical parameters
- Real-time mortality probability prediction
- Color-coded risk assessment
- SHAP force plot for feature impact visualization
- Waterfall plot showing feature contribution hierarchy
""")
st.markdown("---")

# ==========================
# Feature Definitions
# ==========================

# List of all clinical features used in the model
FEATURES = ['D-dimer', 'AG', 'Age', 'NIHSS', 'Time', 'WBC', 'NSE', 'Glucose']

# Descriptions of each feature with units and clinical significance
FEATURE_DESCRIPTIONS = {
    'D-dimer': 'D-dimer (μg/mL) - Protein fragment produced by blood clot breakdown, marker of thrombosis',
    'AG': 'Anion Gap (mmol/L) - Difference between serum cations and anions, indicator of acid-base balance',
    'Age': 'Patient age in years',
    'NIHSS': 'NIH Stroke Scale Score (0-42) - Measures severity of neurological deficits',
    'Time': 'Time from symptom onset to hospital admission (hours)',
    'WBC': 'White Blood Cell count (×10⁹/L) - Immune system activity marker',
    'NSE': 'Neuron-Specific Enolase (ng/mL) - Enzyme released during neuronal damage',
    'Glucose': 'Fasting blood glucose level (mmol/L) - Metabolic marker'
}

# Configuration for input widgets (type, range, default value, step, formatting)
FEATURE_CONFIGS = {
    'D-dimer':  dict(kind='float', min=0.1,  max=20.0, value=0.99,  step=0.1, fmt="%.2f"),  # Thrombosis marker
    'AG':       dict(kind='float', min=0.0,  max=25.0, value=2.0,   step=0.1, fmt="%.2f"),  # Acid-base balance
    'Age':      dict(kind='int',   min=18,   max=100,  value=81,    step=1),              # Age in years
    'NIHSS':    dict(kind='int',   min=0,    max=42,   value=24,    step=1),              # Neurological deficit
    'Time':     dict(kind='int',   min=0,    max=72,   value=1,     step=1),              # Onset to admission
    'WBC':      dict(kind='float', min=0.0,  max=30.0, value=19.22, step=0.1, fmt="%.2f"),  # Immune response
    'NSE':      dict(kind='float', min=0.0,  max=100.0,value=9.57,  step=0.1, fmt="%.2f"),  # Brain injury
    'Glucose':  dict(kind='float', min=0.0,  max=30.0, value=5.88,  step=0.1, fmt="%.2f")   # Metabolic marker
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
        '/home/Jingle/data/model/朱正保/death_model.pkl',
        './death_model.pkl',
        './model/death_model.pkl'
    ]
    
    scaler_paths = [
        '/home/Jingle/data/model/朱正保/scaler_model.pkl',
        './scaler_model.pkl',
        './model/scaler_model.pkl'
    ]
    
    # Load model
    mdl = None
    for path in model_paths:
        try:
            mdl = joblib.load(path)["model"]
            if mdl:
                st.success(f"Loaded model from: {path}")
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
                st.success(f"Loaded scaler from: {path}")
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

# Sidebar: Input controls
st.sidebar.header("Input Clinical Features")

# Add a reset button using session state
if 'reset' not in st.session_state:
    st.session_state.reset = False

if st.sidebar.button('Reset Inputs'):
    st.session_state.reset = True
    st.experimental_rerun()

# Create two columns in the sidebar for better organization
# Set a wider gap between columns to increase effective sidebar width
col1, col2 = st.sidebar.columns([5, 5], gap="large")

input_values = {}

# Group features into logical categories
laboratory_features = ['D-dimer', 'AG', 'WBC', 'NSE', 'Glucose']
clinical_features = ['Age', 'NIHSS', 'Time']

# Add section headers
st.sidebar.markdown("### Laboratory Results")
for feature in laboratory_features:
    cfg = FEATURE_CONFIGS[feature]
    label = f"{feature} ({FEATURE_DESCRIPTIONS[feature].split(' - ')[0]})"
    
    # Determine which column to place the feature in
    with (col1 if feature in ['D-dimer', 'WBC', 'Glucose'] else col2):
        if cfg['kind'] == 'float':
            input_values[feature] = st.number_input(
                label=label,
                min_value=float(cfg['min']),
                max_value=float(cfg['max']),
                value=float(cfg['value']) if not st.session_state.reset else float(cfg['min']),
                step=float(cfg['step']),
                format=cfg.get('fmt', "%.2f"),
                help=FEATURE_DESCRIPTIONS[feature]
            )
        else:
            input_values[feature] = st.number_input(
                label=label,
                min_value=int(cfg['min']),
                max_value=int(cfg['max']),
                value=int(cfg['value']) if not st.session_state.reset else int(cfg['min']),
                step=int(cfg['step']),
                help=FEATURE_DESCRIPTIONS[feature]
            )

st.sidebar.markdown("### Clinical Information")
for feature in clinical_features:
    cfg = FEATURE_CONFIGS[feature]
    label = f"{feature} ({FEATURE_DESCRIPTIONS[feature].split(' - ')[0]})"
    
    with (col1 if feature in ['Age'] else col2):
        if cfg['kind'] == 'float':
            input_values[feature] = st.number_input(
                label=label,
                min_value=float(cfg['min']),
                max_value=float(cfg['max']),
                value=float(cfg['value']) if not st.session_state.reset else float(cfg['min']),
                step=float(cfg['step']),
                format=cfg.get('fmt', "%.2f"),
                help=FEATURE_DESCRIPTIONS[feature]
            )
        else:
            input_values[feature] = st.number_input(
                label=label,
                min_value=int(cfg['min']),
                max_value=int(cfg['max']),
                value=int(cfg['value']) if not st.session_state.reset else int(cfg['min']),
                step=int(cfg['step']),
                help=FEATURE_DESCRIPTIONS[feature]
            )

# Reset the reset flag after processing
if st.session_state.reset:
    st.session_state.reset = False

# Create DataFrame from input values
sample_df = pd.DataFrame([input_values], columns=FEATURES)

# Load model and scaler
model, scaler = load_model_and_scaler()
if model is None:
    st.stop()

# Use a spinner to show processing status
with st.spinner('Processing input features and generating prediction...'):
    # Preprocess features for prediction
    X_model_df = preprocess_features(sample_df, model, scaler)
    
    # Generate SHAP explanations
    base_value, shap_used = generate_shap_explanations(model, X_model_df, sample_df)

# Make prediction and display results
st.subheader("Prediction Result")

# Create a container for the prediction results
result_container = st.container()

with result_container:
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
        
        # Display results with improved layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Mortality Probability:** {mortality_prob:.1f}%")
            progress_bar = st.progress(mortality_prob / 100.0)
        
        with col2:
            # Determine risk level based on probability
            if mortality_prob < 20:
                risk, color = "Low Risk", "#28a745"  # Bootstrap green
            elif mortality_prob < 50:
                risk, color = "Medium Risk", "#ffc107"  # Bootstrap yellow
            else:
                risk, color = "High Risk", "#dc3545"  # Bootstrap red
            
            st.markdown(f"**Risk Level:** <span style='color:{color}; font-weight:bold; font-size: 1.2em;'>{risk}</span>", unsafe_allow_html=True)
        
        # Add a results summary
        st.markdown("---")
        st.subheader("Results Summary")
        
        # Create a feature impact summary
        st.markdown("### Key Feature Contributions")
        
        # Get top positive and negative features
        feature_contributions = dict(zip(FEATURES, shap_used[0, :]))
        sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Display top 3 influential features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Increasing Risk:**")
            for feature, contribution in [f for f in sorted_features if f[1] > 0][:3]:
                if contribution > 0:
                    st.markdown(f"- {feature}: +{contribution:.3f}")
        
        with col2:
            st.markdown("**Decreasing Risk:**")
            for feature, contribution in [f for f in sorted_features if f[1] < 0][:3]:
                if contribution < 0:
                    st.markdown(f"- {feature}: {contribution:.3f}")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# ==========================
# SHAP Explanations Section
# ==========================

# Add a divider and section header for clarity
st.markdown("---")
st.subheader("Model Explanation (SHAP Values)")

# Explain what SHAP values represent with clear bullet points
st.markdown("**SHAP values show how each feature influences the prediction:**")
st.markdown("- Red bars/areas: Features that increase mortality risk")
st.markdown("- Blue bars/areas: Features that decrease mortality risk")
st.markdown("- Length/intensity: Magnitude of feature's influence")

# Display SHAP force plot - shows feature contributions for this specific prediction
try:
    # Validate that SHAP values match the number of features
    if shap_used.shape[1] != len(FEATURES):
        st.warning(f"SHAP values shape ({shap_used.shape[1]}) doesn't match feature count ({len(FEATURES)}). Using available features.")
        
    # Create and display the SHAP force plot
    # This plot shows how each feature pushes the prediction from the base value
    st_shap(shap.force_plot(
        base_value=base_value,                     # Average model output
        shap_values=shap_used[0, :],               # SHAP values for current prediction
        features=sample_df.iloc[0, :],             # Actual feature values
        feature_names=FEATURES,                    # Feature names for display
        show=False                                 # Don't show plot immediately
    ), height=220)
    
    # Display waterfall plot - shows feature contributions in descending order
    st.subheader("Waterfall Plot")
    
    # Fix font display issues for negative values
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create a new figure with smaller size and constrained layout to avoid warnings
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    
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
        # max_display: Shows up to 10 most influential features
        shap.plots.waterfall(
            shap_explanation,
            show=False,
            max_display=10
        )
        
        # Display the plot in Streamlit
        st.pyplot(fig)
    
    except Exception as inner_e:
        # Handle errors specific to waterfall plot generation
        st.error(f"Waterfall plot failed: {inner_e}")
    finally:
        # Explicitly close the figure to free memory
        plt.close(fig)
        
# Handle general errors in SHAP visualization
except Exception as e:
    st.error(f"Failed to display SHAP visualizations: {e}")
    # Provide helpful information for troubleshooting
    st.info("For detailed model explanation, ensure SHAP is properly installed and compatible with your model.")

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
