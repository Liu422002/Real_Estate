import streamlit as st
import pandas as pd
import pickle

st.title("Real Estate Price Prediction App")
st.write("Enter the property details below to predict the house price.")

# choose model
model_choice = st.selectbox(
    "Choose a model",
    ["Linear Regression", "Random Forest"]
)

# load model
if model_choice == "Linear Regression":
    model_path = "models/linear_model.pkl"
else:
    model_path = "models/random_forest_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

# user inputs
year_sold = st.number_input("Year Sold", min_value=1900, max_value=2100, value=2013)
property_tax = st.number_input("Property Tax", min_value=0, value=234)
insurance = st.number_input("Insurance", min_value=0, value=81)
beds = st.number_input("Beds", min_value=0, value=1)
baths = st.number_input("Baths", min_value=0, value=1)
sqft = st.number_input("Square Feet", min_value=0, value=584)
year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=2013)
lot_size = st.number_input("Lot Size", min_value=0, value=0)

basement = st.selectbox("Basement", [0, 1])
popular = st.selectbox("Popular Area", [0, 1])
recession = st.selectbox("Recession", [0, 1])

property_age = st.number_input("Property Age", min_value=0, value=0)
property_type_condo = st.selectbox("Property Type Condo", [0, 1])

# prediction button
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "year_sold": year_sold,
        "property_tax": property_tax,
        "insurance": insurance,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year_built,
        "lot_size": lot_size,
        "basement": basement,
        "popular": popular,
        "recession": recession,
        "property_age": property_age,
        "property_type_Condo": property_type_condo
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Price: ${prediction:,.2f}")