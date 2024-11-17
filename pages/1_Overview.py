import streamlit as st

st.subheader("Project Overview")
st.write("""
    This project predicts house prices based on several key features. The model allows users to select 
    different machine learning algorithms and test their performance. It provides insights into the 
    importance of various features in determining house prices and helps users understand how models make predictions.
""")

st.subheader("Dataset Overview")
st.write("""
    - The dataset contains 13 features collected from a real estate dataset:
        1. **Area (Square Feet)**: Total area of the house in square feet.
        2. **Number of Bedrooms**: Count of bedrooms in the house.
        3. **Number of Bathrooms**: Count of bathrooms in the house.
        4. **Number of Stories**: The number of floors in the house.
        5. **Main Road**: Whether the house is located near the main road (Yes/No).
        6. **Guest Room**: Whether the house has a guest room (Yes/No).
        7. **Basement**: Whether the house has a basement (Yes/No).
        8. **Hot Water Heating**: Availability of hot water heating (Yes/No).
        9. **Air Conditioning**: Availability of air conditioning (Yes/No).
        10. **Parking Spaces**: Number of parking spaces available.
        11. **Preferred Area**: Whether the house is located in a preferred area (Yes/No).
        12. **Furnishing Status**: Status of furnishing (Furnished, Semi-Furnished, or Unfurnished).
    - The target variable is the **house price**, measured in dollars.
""")

st.subheader("Exploratory Data Analysis")
st.write("""
    Key insights from the initial data exploration include:
    - The **area of the house** is positively correlated with the price; larger houses tend to be more expensive.
    - Houses with **more bathrooms and bedrooms** generally have higher prices.
    - Features like **air conditioning**, **guest rooms**, and **preferred area** significantly influence the price.
    - The dataset shows some skewness in house prices, with a few very expensive outliers.
""")

st.subheader("Modeling Overview")
st.write("""
    - The app supports multiple models, including:
        - **Linear Regression**: Simple and interpretable but less robust to non-linear relationships.
        - **Random Forest**: A tree-based ensemble model that captures non-linear interactions effectively.
    - Model performance was evaluated using metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    - Average residuals across models: $785,290
""")

st.subheader("Feature Importance")
st.write("""
    - The app includes a visualization of feature importance, helping users understand which features contribute the most to predictions.
    - For **Linear Regression**, importance is determined by the absolute values of the coefficients.
    - For **Random Forest**, importance is determined by the contribution of each feature to the reduction in impurity.
""")