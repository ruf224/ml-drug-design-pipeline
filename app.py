import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="CADD Screening App", page_icon="🧪", layout="wide")

# Sidebar Workflow Navigation UI Component
st.sidebar.title("🧬 Navigation Terminal")
steps = [
    "1. Overview & Setup Check",
    "2. Data Upload & Chemical Features",
    "3. Machine Learning Evaluation",
    "4. ADMET Filtering Stage",
    "5. Final Wet-Lab Recommendation"
]
current_step = st.sidebar.radio("Follow the phases sequentially:", steps)

# Secure multi-user variable isolation using Streamlit Session States
if 'compounds_df' not in st.session_state:
    st.session_state.compounds_df = None
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

@st.cache_resource
def fetch_validated_model():
    if os.path.exists('drug_classifier_model.pkl'):
        with open('drug_classifier_model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

ml_model = fetch_validated_model()

# --- PHASE 1: OVERVIEW ---
if current_step == "1. Overview & Setup Check":
    st.title("🧪 Smart Screening & Chemical Evaluation Pipeline")
    st.info("This system assists research teams by executing binary molecular activity classifiers before launching chemical asset synthesis.")
    if ml_model is not None:
        st.success("🟢 Success: Trained Model Artifact (`drug_classifier_model.pkl`) detected and active.")
    else:
        st.warning("⚠️ Warning: Model artifact not found in repository. App is operating in placeholder evaluation configuration.")

# --- PHASE 2: CHEMICAL DESCRIPTORS INPUT ---
elif current_step == "2. Data Upload & Chemical Features":
    st.title("📊 Phase 2: Configure Candidate Chemical Specifications")
    mode = st.radio("Select entry type:", ["Single Molecule Manual Entry", "Batch Matrix Simulation"])
    
    if mode == "Single Molecule Manual Entry":
        c1, c2 = st.columns(2)
        with c1:
            mw = st.number_input("Molecular Weight", 100.0, 1000.0, 380.0)
            logp = st.number_input("LogP Coefficient", -2.0, 8.0, 2.8)
        with c2:
            hbd = st.number_input("H-Bond Donors", 0, 15, 2)
            hba = st.number_input("H-Bond Acceptors", 0, 20, 5)
        if st.button("Construct Compound Record"):
            st.session_state.compounds_df = pd.DataFrame([{
                'Molecule_ID': 'MOL_USER_01', 'Molecular_Weight': mw, 'LogP': logp, 'H_Bond_Donors': hbd, 'H_Bond_Acceptors': hba
            }])
            st.session_state.predictions_made = False
            st.success("Record constructed! Proceed to Phase 3.")
            
    else:
        num = st.slider("Select test size matrix:", 2, 10, 4)
        if st.button("Generate Matrix Batch"):
            st.session_state.compounds_df = pd.DataFrame({
                'Molecule_ID': [f'BATCH_MOL_{i+1}' for i in range(num)],
                'Molecular_Weight': np.round(np.random.uniform(220, 580, num), 2),
                'LogP': np.round(np.random.uniform(0.5, 5.5, num), 2),
                'H_Bond_Donors': np.random.randint(0, 6, num),
                'H_Bond_Acceptors': np.random.randint(1, 10, num)
            })
            st.session_state.predictions_made = False
            st.success("Batch constructed! Proceed to Phase 3.")

    if st.session_state.compounds_df is not None:
        st.dataframe(st.session_state.compounds_df, use_container_width=True)

# --- PHASE 3: ML EVALUATION ENGINE ---
elif current_step == "3. Machine Learning Evaluation":
    st.title("🤖 Phase 3: ML Bioactivity Probability Engine")
    if st.session_state.compounds_df is None:
        st.error("❌ No chemical data structures detected. Return to Phase 2.")
    else:
        if st.button("Execute Predictive Assessment"):
            feats = st.session_state.compounds_df[['Molecular_Weight', 'LogP', 'H_Bond_Donors', 'H_Bond_Acceptors']]
            if ml_model is not None:
                probs = ml_model.predict_proba(feats)[:, 1]
                preds = ml_model.predict(feats)
            else:
                # Math proxy configuration baseline fallback
                probs = np.round(1 / (1 + np.exp(-(feats['LogP'] - 2.0))), 3)
                preds = (probs >= 0.5).astype(int)
                
            res = st.session_state.compounds_df.copy()
            res['Bioactivity_Probability'] = probs
            res['Prediction'] = np.where(preds == 1, "🟢 ACTIVE", "🔴 INACTIVE")
            st.session_state.compounds_df = res
            st.session_state.predictions_made = True
            st.success("Classification complete!")
            
        if st.session_state.predictions_made:
            st.dataframe(st.session_state.compounds_df, use_container_width=True)

# --- PHASE 4: ADMET FILTER ---
elif current_step == "4. ADMET Filtering Stage":
    st.title("🛡️ Phase 4: Absorption Safety Boundaries Screen")
    if not st.session_state.predictions_made:
        st.error("❌ Execute Phase 3 Machine Learning processing before checking safety limits.")
    else:
        max_logp_thresh = st.slider("Set Maximum Safe Drug Lipophilicity Threshold (LogP Boundary):", 3.0, 6.0, 5.0)
        df_safety = st.session_state.compounds_df.copy()
        
        # Enforce Lipinski metrics boundary
        df_safety['ADMET_Pass'] = (df_safety['Molecular_Weight'] <= 500) & (df_safety['LogP'] <= max_logp_thresh)
        df_safety['ADMET_Status'] = np.where(df_safety['ADMET_Pass'], "✅ SAFE PASS", "⚠️ TOXIC/ABSORPTION RISK")
        
        st.session_state.compounds_df = df_safety
        st.dataframe(st.session_state.compounds_df, use_container_width=True)

# --- PHASE 5: WET-LAB RECOMMENDATION VERDICT ---
elif current_step == "5. Final Wet-Lab Recommendation":
    st.title("🚀 Phase 5: Pipeline Verdict & Assay Synthesis Recommendation")
    if st.session_state.compounds_df is None or 'ADMET_Status' not in st.session_state.compounds_df.columns:
        st.error("❌ Process all previous stages before calculating chemical assay recommendations.")
    else:
        df_fin = st.session_state.compounds_df.copy()
        
        # Master selection rubric: Active evaluation AND passes metabolic absorption tests
        df_fin['Wet_Lab_Verdict'] = np.where(
            (df_fin['Prediction'] == "🟢 ACTIVE") & (df_fin['ADMET_Status'] == "✅ SAFE PASS"),
            "🚀 GO: Proceed to Assay/Wet-Lab",
            "🛑 HOLD: Reject / Structure Optimization Required"
        )
        
        st.dataframe(df_fin[['Molecule_ID', 'Prediction', 'ADMET_Status', 'Wet_Lab_Verdict']], use_container_width=True)
        
        go_total = int(np.sum(df_fin['Wet_Lab_Verdict'] == "🚀 GO: Proceed to Assay/Wet-Lab"))
        st.metric(label="Total Structures Recommended for Synthesis", value=f"{go_total} / {len(df_fin)}")
