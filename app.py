from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st
from PIL import Image



# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Split into features and label
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model accuracy
training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))

# Streamlit web app
st.title('❤️ Heart Disease Prediction App')

# Load and display image
img = Image.open('heart_img.jpg')
st.image(img, width=150)

st.subheader('🔍 Enter Patient Data')

# Labels for each input field
input_labels = [
    "Age",
    "Sex (1 = male, 0 = female)",
    "Chest Pain Type (0-3)",
    "Resting Blood Pressure",
    "Cholesterol",
    "Fasting Blood Sugar (1 = true, 0 = false)",
    "Resting ECG Results (0-2)",
    "Max Heart Rate Achieved",
    "Exercise Induced Angina (1 = yes, 0 = no)",
    "Oldpeak",
    "Slope (0-2)",
    "Number of Major Vessels (0-3)",
    "Thal (1 = normal, 2 = fixed defect, 3 = reversable defect)"
]

# Create input fields
user_input = []
for label in input_labels:
    value = st.number_input(label, value=0.0)
    user_input.append(value)

# Predict heart disease when button is pressed
if st.button("Predict"):
    np_df = np.asarray(user_input, dtype=float).reshape(1, -1)
    prediction = model.predict(np_df)

    st.subheader("🧠 Prediction Result:")
    if prediction[0] == 0:
        st.success("✅ The person is unlikely to have heart disease.")
    else:
        st.error("⚠️ The person may have heart disease.")

    st.write("🔢 Raw prediction:", int(prediction[0]))

# Show dataset and model accuracy
st.subheader("📊 Sample of the Dataset")
st.write(heart_data.head())

st.subheader("✅ Model Performance")
st.write(f"Training Accuracy: **{training_data_accuracy:.2f}**")
st.write(f"Test Accuracy: **{test_data_accuracy:.2f}**")

from sklearn.metrics import confusion_matrix, classification_report

st.subheader("🧪 Classification Report (Test Data)")
predictions = model.predict(X_test)
st.text(classification_report(Y_test, predictions))
