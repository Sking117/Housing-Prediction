import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset (Replace 'housing_data.csv' with actual dataset path)
df = pd.read_csv('housing_data.csv')

# Preprocess the data (handle missing values, encode categorical variables, etc.)
df.fillna(df.median(), inplace=True)
X = df.drop(columns=['Price'])  # Assuming 'Price' is the target variable
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'housing_price_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model Evaluation: MAE={mae}, MSE={mse}")

# Streamlit Web Application
def main():
    st.title("Housing Price Prediction")
    st.write("Enter the details below to predict the housing price:")
    
    inputs = {}
    for col in X.columns:
        inputs[col] = st.number_input(f"Enter {col}", value=0.0)
    
    if st.button("Predict Price"):
        model = joblib.load('housing_price_model.pkl')
        input_data = np.array([list(inputs.values())]).reshape(1, -1)
        prediction = model.predict(input_data)
        st.write(f"Predicted Housing Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()

