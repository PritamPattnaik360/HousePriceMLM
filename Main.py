import streamlit as st
import webbrowser

st.subheader("What is this?")
st.write("""
    This project predicts house prices using machine learning models, trained on a dataset containing features like area, number of bedrooms, and bathrooms. 
    In Jupyter Notebook, I preprocessed the data (handling missing values, encoding categorical features), split it into training and testing sets, and trained 
    models like Linear Regression, Random Forest, Decision Tree, and Neural Network, saving them with joblib for deployment
""")
st.write("""
    You can find a nice overview of this project on the overview page.
    When on the testing page when you predict house prices based on the model chosen you can see how the model
    weights each factor given from the dataset.
""")
st.subheader("Learn more")
st.write("""
    Check out my github to see the specific jupiter notebook file on how I trained the model.
""")
st.markdown("[My Git Repo](https://github.com/PritamPattnaik360/HousePriceMLM)")

