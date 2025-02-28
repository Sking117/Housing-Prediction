import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data_path = 'housing_data.csv'
df = pd.read_csv(data_path)
print(df.head())  # Ensure data is loaded correctly



# Preprocessing
def preprocess_data(df):
    df = df.dropna()  # Handle missing values by dropping (can be improved)
    df.fillna(df.mode().iloc[0], inplace=True)
    X = df.drop(columns=['SalePrice'])  # Features (Assuming 'price' is the target variable)
    y = df['SalePrice']  # Target
    
    # Identify numerical and categorical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = X.select_dtypes(include=['object']).columns
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    return X, y, preprocessor

X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model training
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "housing_price_model.pkl")

# Streamlit App
def main():
    st.title("Housing Price Prediction App")
    
    # Get user input
    input_data = {}
    for col in X.columns:
        if col in df.select_dtypes(include=['object']).columns:
            input_data[col] = st.selectbox(f"Select {col}", df[col].unique())
        else:
            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].median()))
    
    if st.button("Predict Price"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.write(f"Predicted Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()

# Deployment steps:
# 1. Save this script and the model file in a GitHub repository.
# 2. Create a requirements.txt file with necessary libraries.
# 3. Deploy on Streamlit Cloud using the repository URL.


