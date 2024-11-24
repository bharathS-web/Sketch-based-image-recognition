import os
import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt


# Model path
model_path = 'model.keras'

# Check if the model file exists
if os.path.exists(model_path):
    try:
        # Load Model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    st.error(f"Model file not found: {model_path}")
    model = None

f = open("categories.txt","r")
classes = f.readlines()
f.close()

class_names = [c.replace('\n','').replace(' ','_') for c in classes]

# Initialize the drawing board dimensions
WIDTH, HEIGHT = 28, 28

def predict_drawing(image, top_k=5):
    """Predict the drawing using the model."""
    pred = model.predict(image)[0]
    top_indices = (-pred).argsort()[:top_k]
    return [(class_names[i], pred[i]) for i in top_indices]

# Streamlit UI
st.title("Quick Draw Game ✈️")
st.write("Draw something on the board...")

# Initialize session state for canvas and prediction result
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Create two columns for layout
col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed

# Create a canvas for drawing in the first column
with col1:
    canvas_result = st_canvas(
        fill_color="black",  # Background color
        stroke_color="white",  # Drawing color
        stroke_width=10,
        height=HEIGHT * 10,
        width=WIDTH * 10,
        key="canvas",
        drawing_mode="freedraw",
        update_streamlit=True
    )

    # Store the canvas data in session state
    if canvas_result.image_data is not None:
        st.session_state.canvas_data = canvas_result.image_data

# Create the prediction area in the second column
with col2:
    # Predict button
    
    if True :
        if model is not None and st.session_state.canvas_data is not None:
            # Convert the canvas to a NumPy array
            image = np.array(st.session_state.canvas_data)
            
            # Convert the image to grayscale and normalize
            image = np.mean(image, axis=2) / 255.0  # Convert to grayscale and normalize
            
            # Resize the image to 28x28
            image = Image.fromarray((image * 255).astype(np.uint8))  # Convert back to uint8 for resizing
            image = image.resize((WIDTH, HEIGHT))
            image = np.array(image) / 255.0  # Normalize again after resizing
            
            # Reshape for the model: (1, 28, 28, 1)
            image = image.reshape((1, HEIGHT, WIDTH, 1))  # Add channel dimension
            
            # Make prediction
            predicted_value = predict_drawing(image)
            st.session_state.prediction_result = predicted_value
        else:
            st.write("Please draw something before predicting.")

    # Display prediction result
    if st.session_state.prediction_result is not None:
        st.write("Predicted Classes and Probabilities:")
        labels = [class_name for class_name, _ in st.session_state.prediction_result]
        probabilities = [probability for _, probability in st.session_state.prediction_result]

        for class_name, probability in st.session_state.prediction_result:
            st.write(f"{class_name}: {probability:.2f}")

        # Create a pie chart
        fig, ax = plt.subplots()
        ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the pie chart in Streamlit
        st.pyplot(fig)
