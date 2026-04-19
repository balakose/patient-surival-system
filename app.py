import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Page Config ---
st.set_page_config(page_title="Patient Survival Analysis & Prediction", layout="wide")

# --- 2. Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Analysis Dashboard", "Survival Prediction"])

# --- 3. Load Data & Model ---
@st.cache_resource
def load_data():
    # Hum khud column names de rahe hain
    column_names = ['age', 'operational_year', 'axillary_nodes', 'survival_status']
    
    # 'names' parameter use karne se Pandas pehli row ko data maanega, header nahi
    df = pd.read_csv('haberman.csv', names=column_names)
    return df

@st.cache_resource
def load_model():
    # Make sure scaler and model are in your GitHub main folder
    model = pickle.load(open('model.pkl', 'rb'))
    # Hum dummy scaler bana rahe hain agar aapke paas scaler.pkl nahi hai
    # Lekin actual scaler.pkl ho toh use load karna best hai
    return model

# --- PAGE 1: DATA ANALYSIS ---
if page == "Data Analysis Dashboard":
    st.title("📊 Patient Data Insights (Haberman's Dataset)")
    
    try:
        df = load_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            # Sirf numerical columns ka correlation
            corr = df.corr()
            sns.heatmap(corr, annot=True, ax=ax, cmap='plasma')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Operational Year Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df['operational_year'], kde=True, ax=ax2, color='olive')
            st.pyplot(fig2)
            
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"Error loading CSV file: {e}. Make sure 'haberman.csv' is on GitHub.")

# --- PAGE 2: PREDICTION ---
else:
    st.title("🏥 Patient Survival Prediction")
    st.markdown("Enter clinical details to predict if the patient survives more than 5 years.")

    model = load_model()
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 0, 100, 30)
        year = c2.number_input("Year of Operation (60s)", 0, 99, 64)
        nodes = c3.number_input("Axillary Nodes", 0, 100, 0)
        
        submit = st.form_submit_button("Predict Result")

    if submit:
        # Simple prediction (Add scaling if you used it in training)
        features = np.array([[age, year, nodes]])
        pred = model.predict(features)
        
        st.divider()
        if pred[0] == 1:
            st.success("✅ Prediction: Patient is likely to survive 5 years or longer.")
        else:
            st.error("⚠️ Prediction: Patient may not survive more than 5 years.")
