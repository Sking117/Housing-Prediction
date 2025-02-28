import streamlit as st
import pandas as pd

# Define `data` before using it
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Now you can use `data`
st.write(data)
# Preprocess Data (Handle missing values and categorical variables if necessary)
data = data.dropna()
X = data.drop(columns=['Price'])
y = data['Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Performance: MAE = {mae}, MSE = {mse}")
st.write(f"Model Performance: MAE = {mae}, MSE = {mse}")

# Save the trained model
joblib.dump(model, "housing_price_model.pkl")

# Streamlit Web Application
def predict_price(features):
    model = joblib.load("housing_price_model.pkl")
    prediction = model.predict([features])
    return prediction[0]

st.title("Housing Price Prediction")
st.write("Enter the details to predict the housing price:")

# Creating input fields dynamically based on features
features = []
for col in X.columns:
    val = st.number_input(f"Enter {col}", value=0.0)
    features.append(val)

if st.button("Predict Price"):
    predicted_price = predict_price(features)
    print(f"Predicted Housing Price: ${predicted_price:.2f}")
    st.write(f"Predicted Housing Price: ${predicted_price:.2f}")

# Instructions to Save and Deploy on GitHub
st.write("### Steps to Deploy:")
st.write("1. Save this script as `app.py`.")
st.write("2. Push the script to a GitHub repository.")
st.write("3. Deploy using Streamlit by running `streamlit run app.py`.")
