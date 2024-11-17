import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


feature_columns = joblib.load('feature_columns.pkl')

st.subheader("Select a Model")
model_options = {
    "Linear Regression": "house_price_model_lr.pkl",
    "Random Forest": "house_price_model_rf.pkl"
}
selected_model_name = st.selectbox("Choose a model to test", options=list(model_options.keys()))

selected_model_file = model_options[selected_model_name]
model = joblib.load(selected_model_file)

st.title("House Price Prediction Linear Regression and Random Forest Model")


square_feet = st.number_input("Square Feet", min_value=500, max_value=10000, step=10, value=1000, key="sq_ft")
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1, value=3, key="bedrooms")
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, value=1, key="bathrooms")

data = {
    'area': [square_feet],
    'bedrooms': [num_bedrooms],
    'bathrooms': [num_bathrooms],
}
input_df = pd.DataFrame(data)


input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df.reindex(columns=feature_columns, fill_value=0)


def plot_feature_importance(model, feature_columns, model_type):
    if model_type == "Linear Regression":
        importance = model.coef_[0] if hasattr(model, "coef_") else model.coef_
        feature_importance = pd.Series(importance, index=feature_columns).sort_values(key=abs, ascending=False)
    elif model_type == "Random Forest":
        importance = model.feature_importances_
        feature_importance = pd.Series(importance, index=feature_columns).sort_values(ascending=False)
    else:
        st.write("Feature importance not supported for this model type.")
        return None

    plt.figure(figsize=(8, 6))
    feature_importance.plot(kind="bar")
    plt.title(f"{model_type} Feature Importance")
    plt.ylabel("Importance Score")
    plt.xlabel("Feature")
    st.pyplot(plt.gcf())  

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader(f"Estimated House Price: ${prediction[0]:,.2f}")
    plot_feature_importance(model, feature_columns, selected_model_name)
    st.write(""" 
        The Graph displayed shows the weighting of imprtance each specific trait has on the overall number
    """)

