import os
import json
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st

excel_path = "/content/pesticide data.xlsx"

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_path)

# Create a dictionary to map diseases to pesticides
disease_to_pesticide = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

# Function to get the pesticide for a given disease
def get_pesticide(disease):
    return disease_to_pesticide.get(disease, "No pesticide found for this disease")


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"/content/drive/MyDrive/new model plant/plant_disease_prediction_model.h5"
# Load the pre-trained models
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"/content/drive/MyDrive/plant-disease-prediction-cnn-deep-leanring-project-main/app/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = predictions[0][predicted_class_index]*100
    return predicted_class_name,confidence


# Streamlit App
st.title('AGRASCAN: Plant Disease Detection App')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction,confidence = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)} with {confidence:.2f}% confidence')
            
            # Get the corresponding pesticide
            pesticide = get_pesticide(prediction)
            st.info(f'Recommended Pesticide: {pesticide}')