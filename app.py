
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered"
)

# Load trained models
@st.cache_resource
def load_models():
    rf_model = joblib.load("random_forest_model.pkl")

    xgb_model = XGBRegressor()
    xgb_model.load_model("xgboost_model.json")

    preprocessor = joblib.load("preprocessor.pkl")

    return rf_model, xgb_model, preprocessor

rf_model, xgb_model, preprocessor = load_models()

# App title
st.title(" Lahore House Price Predictor")
st.write("Enter your property details to estimate its price.")

# Get available categories
encoder = preprocessor.named_transformers_["cat"]

property_types = list(encoder.categories_[0])
locations = list(encoder.categories_[1])

# User inputs
st.subheader("Property Details")

property_type = st.selectbox(
    "Property Type", property_types
)

location = st.selectbox(
    "Location", sorted(locations)
)

area = st.number_input(
    "Area (Marla)",
    min_value=1.0,
    max_value=182.0,
    value=10.0,
    step=0.5
)

baths = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=3
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=11,
    value=3
)

model_choice = st.selectbox(
    "Select Prediction Model",
    ["Random Forest", "XGBoost"]
)

# Prediction
if st.button("Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([{
        "Type": property_type,
        "Location": location,
        "Area": area,
        "Bath(s)": baths,
        "Bedroom(s)": bedrooms
    }])

    input_encoded = preprocessor.transform(input_data)

    if model_choice == "Random Forest":
        predicted_price = rf_model.predict(input_encoded)[0]
    else:
        predicted_price = xgb_model.predict(input_encoded)[0]

    st.success("Price Prediction Completed!")

    st.subheader("Estimated House Price")

    st.markdown(f"## PKR {predicted_price:,.0f}")

    st.write(
        f"Approximately PKR {predicted_price / 10000000:.2f} Crore"
    )

    st.caption(
        "This is an estimated price based on the trained machine learning model, not a guaranteed market price."
    )
