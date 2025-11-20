import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Saudi Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ----------------- LOAD MODEL + COLUMNS -----------------
model = joblib.load("random_forest.pkl")
model_columns = joblib.load("model_columns.pkl")  

# ----------------- HEADER -----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#0E4D92;'>🚗 Saudi Used Car Price Predictor</h1>
    <p style='text-align:center; font-size:18px; color:#555;'>Enter your car's details and get an instant AI-powered price estimation.</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.header("📝 Car Information")
st.sidebar.write("Provide the car specifications below:")

categorical_val = ['Color', 'Options', 'Gear_Type', 'Make']

make = st.sidebar.selectbox("🔰 Make", [
    'Hyundai','Dodge','Toyota','Jeep','Kia','Chevrolet','Volkswagen','Ford','Nissan',
    'GMC','Mercury','Other','Mercedes','Škoda','Honda','Suzuki','INFINITI','BMW',
    'Renault','Chery','Peugeot','Mazda','Geely','Mitsubishi','Lexus','Chrysler',
    'Lincoln','MG','Cadillac','Porsche','Daihatsu','Subaru','Audi','Fiat','FAW',
    'Land Rover','Hummer','Classic','Changan','Lifan','Isuzu','BYD','Victory Auto',
    'Zhengzhou','Jaguar','Foton','Genesis','MINI','GAC','HAVAL','Iveco','Great Wall',
    'Bentley','Maserati','Aston Martin','Ferrari','Rolls-Royce'
])

year = st.sidebar.slider("📅 Model Year", 1990, 2025, 2018)

color = st.sidebar.selectbox("🎨 Color", [
    'Another Color','Grey','Silver','White','Navy','Black','Brown','Orange',
    'Blue','Oily','Green','Yellow','Red','Bronze','Golden'
])

options = st.sidebar.selectbox("⚙️ Options Package", [
    'Standard','Semi Full','Full'
])

gear_type = st.sidebar.selectbox("🕹 Gear Type", ['Automatic', 'Manual'])

engine_size = st.sidebar.number_input("🔧 Engine Size (Liters)", 
                                      min_value=0.5, max_value=8.0, 
                                      value=2.0, step=0.1)

mileage = st.sidebar.number_input("📍 Mileage (KM)", 
                                  min_value=0, max_value=500_000, 
                                  value=120_000, step=5000)


# ----------------- BUILD INPUT DATAFRAME -----------------
input_dict = {
    'Make': [make],
    'Year': [year],
    'Color': [color],
    'Options': [options],
    'Engine_Size': [engine_size],
    'Gear_Type': [gear_type],
    'Mileage': [mileage]
}

input_df = pd.DataFrame(input_dict)

# Create dummies as done during training
dummies_input = pd.get_dummies(input_df[categorical_val])
base = input_df.drop(categorical_val, axis=1)
input_encoded = pd.concat([base, dummies_input], axis=1)

# Align columns with model
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# ----------------- MAIN CONTENT -----------------
left, right = st.columns([1, 2])

with left:
    st.markdown("## 🔍 Prediction")

    if st.button("💰 Predict Price", use_container_width=True):
        price = model.predict(input_encoded.values)[0]


        st.success(
            f"### Estimated Price: **{price:,.0f} SAR**"
        )

        st.markdown(
            "<p style='color:#555;'>This is an AI-generated estimation based on thousands of real listings.</p>",
            unsafe_allow_html=True
        )

with right:
    st.markdown("## 📊 Top 10 Model Feature Importances")

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': model_columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(fi_df['feature'], fi_df['importance'], color="#0E4D92")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features Influencing Price")
    ax.invert_yaxis()
    
    st.pyplot(fig)

st.markdown("---")

# ----------------- FOOTER -----------------
st.markdown(
    """
    <p style='text-align:center; color:#888; font-size:14px;'>
    Developed by Lubna Rama Raghad Sara :D  
    </p>
    """,
    unsafe_allow_html=True
)

