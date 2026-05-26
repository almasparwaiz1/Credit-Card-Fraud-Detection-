import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration & Explicit Paths --- 
BASE_DIR = r"F:\AI and Data Science Projects\house price prediction app\Streamlit_Frontend"

MODEL_PATH = os.path.join(BASE_DIR, 'stacking_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, 'feature_columns.joblib')

# --- Load Resources (with caching for performance) ---
@st.cache_resource
def load_resources():
    """Loads the pre-trained model, scaler, and feature columns safely."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error(f"Error: Required file missing. Please ensure assets are saved inside: {BASE_DIR}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading system assets: {e}")
        st.stop()

model, scaler, feature_columns = load_resources()

# --- Preprocessing Function ---
def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """Applies the feature engineering rules matching training specifications."""
    df_raw_input = pd.DataFrame([input_dict])

    # 1. Interaction Features
    df_raw_input['Amount_V1_Interaction'] = df_raw_input['Amount'] * df_raw_input['V1']
    df_raw_input['Amount_V2_Interaction'] = df_raw_input['Amount'] * df_raw_input['V2']
    df_raw_input['Amount_V3_Interaction'] = df_raw_input['Amount'] * df_raw_input['V3']
    df_raw_input['Amount_V4_Interaction'] = df_raw_input['Amount'] * df_raw_input['V4']

    # 2. Polynomial Features for Amount
    df_raw_input['Amount_squared'] = df_raw_input['Amount']**2
    df_raw_input['Amount_cubed'] = df_raw_input['Amount']**3

    # 3. Ratio Features
    df_raw_input['V1_div_V2'] = df_raw_input['V1'] / (df_raw_input['V2'].replace(0, 1e-6) + 1e-6)

    # Validate feature consistency
    missing_cols = set(feature_columns) - set(df_raw_input.columns)
    if missing_cols:
        st.error(f"Pipeline Mismatch Error: Missing expected architecture columns: {', '.join(missing_cols)}")
        st.stop()
    
    try:
        df_ordered = df_raw_input[feature_columns]
    except KeyError as e:
        st.error(f"Feature sorting failed: Component target '{e}' could not be evaluated.")
        st.stop()

    X_scaled = scaler.transform(df_ordered)
    return pd.DataFrame(X_scaled, columns=feature_columns, index=df_ordered.index)

# --- Prediction Function ---
def make_prediction(processed_data: pd.DataFrame):
    """Calculates evaluation margins against the predictive framework model."""
    prediction_proba = model.predict_proba(processed_data)[:, 1] 
    return prediction_proba[0]

# --- Streamlit Corporate UI Overrides ---
st.set_page_config(
    page_title="Risk Assessment Portal | Credit Fraud Evaluation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional branding injections via Custom CSS rules 
st.markdown("""
    <style>
    /* Base Body Application Colors */
    .stApp {
        background-color: #F8F9FA;
        color: #0A2540;
    }
    
    /* Typography Overrides */
    h1 {
        font-weight: 700 !important;
        color: #0A2540 !important;
        letter-spacing: -0.5px;
    }
    h2, h3, h4 {
        color: #0A2540 !important;
        font-weight: 600 !important;
    }
    
    /* Input Container Optimization */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Metric Enhancement styles */
    div[data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
    }
    
    /* Clean Horizontal Separators */
    hr {
        margin: 1.5rem 0 !important;
        border-color: #CBD5E1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main App Header Layout ---
col_header, col_status = st.columns([3, 1])
with col_header:
    st.title("💳 Transaction Fraud Evaluation Hub")
    st.markdown("Real-time operational risk assessment dashboard tracking behavioral vector signatures.")
with col_status:
    st.caption("System Diagnostics")
    st.success("Analysis Engine Online")

st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.markdown("### Transaction Vector Controls")
st.sidebar.markdown("Manipulate the engineered component values below to simulate live transaction behavior.")

input_data = {}

# Partition inputs layout inside the sidebar for maximum professionalism
with st.sidebar.expander("💳 Main Fiscal Attributes", expanded=True):
    input_data['Amount'] = st.slider('Transaction Value (USD Amount)', 0.01, 25000.0, 150.0, step=0.1, format="$%.2f")

with st.sidebar.expander("🧬 Component Component Signals V1 - V14", expanded=False):
    for i in range(1, 15):
        key = f'V{i}'
        input_data[key] = st.slider(f'Signal Component {key}', -30.0, 30.0, 0.0, step=0.01, format="%.2f")

with st.sidebar.expander("🧬 Component Component Signals V15 - V28", expanded=False):
    for i in range(15, 29):
        key = f'V{i}'
        input_data[key] = st.slider(f'Signal Component {key}', -30.0, 30.0, 0.0, step=0.01, format="%.2f")

# --- Analytical Process Flow ---
processed_input = preprocess_input(input_data)
fraud_probability = make_prediction(processed_input)

# Display Framework Layout
layout_left, layout_right = st.columns([2, 1])

with layout_left:
    st.subheader("Operational Risk Evaluation")
    
    # Establish thresholds and match alerts according to the 10% Accent color constraints
    if fraud_probability >= 0.50:
        # High Risk Alert - Crimson Red Highlight
        st.error(f"🚨 ALERT: CRITICAL RISK DETECTED")
        
        st.metric(
            label="Computed Fraud Threat Level", 
            value=f"{fraud_probability*100:.2f}% Risk Profile", 
            delta="CRITICAL LEVEL EXCEEDED", 
            delta_color="inverse"
        )
        
        st.markdown(f"""
            <div style="background-color: #FFF5F5; padding: 1rem; border-left: 5px solid #DC3545; border-radius: 4px; color: #DC3545; font-weight: 500;">
                <strong>System Action Required:</strong> This profile shows extreme compliance correlation patterns matching baseline fraudulent attributes. Immediate transaction suspension recommended.
            </div>
        """, unsafe_allow_html=True)
        
    elif 0.15 <= fraud_probability < 0.50:
        # Medium Risk Alert - Electric Amber Highlight
        st.warning(f"⚠️ NOTICE: ELEVATED TRANSACTION ANOMALY")
        
        st.metric(
            label="Computed Fraud Threat Level", 
            value=f"{fraud_probability*100:.2f}% Risk Profile", 
            delta="MONITOR AND AUDIT", 
            delta_color="off"
        )
        
        st.markdown(f"""
            <div style="background-color: #FFFDF5; padding: 1rem; border-left: 5px solid #FFBF00; border-radius: 4px; color: #A67C00; font-weight: 500;">
                <strong>Observation Advisory:</strong> Vector metrics reflect mild skew deviations. Secondary verification confirmation protocols advised.
            </div>
        """, unsafe_allow_html=True)
        
    else:
        # Clear / Legitimate Outcome - Professional Deep Navy / Gray Clean execution
        st.success("✅ TRANSACTION VERIFIED SECURE")
        
        st.metric(
            label="Computed Fraud Threat Level", 
            value=f"{fraud_probability*100:.2f}% Risk Profile", 
            delta="SAFE SPECIFICATION COMPLIANT"
        )
        
        st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 1rem; border-left: 5px solid #708090; border-radius: 4px; color: #0A2540;">
                <strong>System Note:</strong> Analytical metrics pass structural sanity profiles. No abnormal transaction indicators observed.
            </div>
        """, unsafe_allow_html=True)

with layout_right:
    st.subheader("Model Insights")
    st.markdown(f"""
        **Pipeline Framework Architecture:**
        * **Ensemble Strategy:** Stacking Regressor Meta-Learner
        * **Active Features Supervised:** {len(feature_columns)} Active Nodes
        * **Data Scaler Mode:** Robust Scaling Standard Matrix
    """)
    
    with st.expander("🔍 Inspection Payload"):
        st.caption("Engineered input profile ready for inference:")
        st.dataframe(processed_input.T.rename(columns={0: "Scaled Value"}), use_container_width=True)

# --- Corporate Footer Layout ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #708090; font-size: 0.85rem;'>"
    "Risk Evaluation Engine v2.4 • Confidential Enterprise Classification Framework"
    "</div>", 
    unsafe_allow_html=True
)