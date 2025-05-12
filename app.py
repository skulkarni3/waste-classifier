import streamlit as st
import torch
from PIL import Image
import io
import os
from efficient_net import EfficientNetWasteClassifier
from waste_classifier import transform, WasteClassifierCNN
import numpy as np

# Set page config
st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .recyclable {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
    }
    .compostable {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
            
    .landfill {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WasteClassifierCNN().to(device)
    
    # Load the best model weights
    model_path = "https://github.com/skulkarni3/waste-classifier/releases/tag/v1.0.0"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    else:
        st.error(f"Model file not found at {model_path}")
        return None

def predict_image(model, image):
    # Transform the image
    image_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    # Get class probabilities
    class_probs = probabilities[0].cpu().numpy()
    
    # Map class indices to names
    class_names = ['Recyclable', 'Compostable', 'Landfill']
    
    return {
        'class': class_names[predicted_class],
        'probabilities': {
            class_names[i]: float(class_probs[i]) * 100
            for i in range(len(class_names))
        }
    }

# App title and description
st.title("♻️ Waste Classifier")
st.markdown("""
    Upload an image or take a photo to classify waste items into:
    - ♻️ Recyclable
    - 🌱 Compostable
    - 🗑️ Landfill
""")

# Load model
model = load_model()
if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Camera input
camera_input = st.camera_input("Or take a photo")

# Process the image
if uploaded_file is not None or camera_input is not None:
    # Get the image from either upload or camera
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    else:
        image = Image.open(io.BytesIO(camera_input.getvalue())).convert('RGB')
    
    # # Display the image
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a classify button
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            # Get prediction
            result = predict_image(model, image)
            prediction_class = result['class'].lower()

            # Define background color and text color based on the prediction class
            if prediction_class == "recyclable":
                background_color = "#2256c7"  # Green for positive
                text_color = "#ffffff"  # White text
            elif prediction_class == "compostable":
                background_color = "#088008"  # Red for negative
                text_color = "#ffffff"  # White text
            else:
                background_color = "#cf3e21"  # Blue for neutral or unknown
                text_color = "#ffffff"  # White text

            # Apply the styles with the chosen background and text colors
            st.markdown(f"""
                <div class="prediction-box {prediction_class}" style="background-color: {background_color};">
                    <h2 style="text-align: center; margin: 0; color: {text_color};">
                        Prediction: {result['class']}
                    </h2>
                </div>
            """, unsafe_allow_html=True)

            # # Display prediction with appropriate styling
            # prediction_class = result['class'].lower()
            # st.markdown(f"""
            #     <div class="prediction-box {prediction_class}">
            #         <h2 style="text-align: center; margin: 0;">Prediction: {result['class']}</h2>
            #     </div>
            # """, unsafe_allow_html=True)
            
            # Display probabilities
            st.subheader("Confidence Scores")
            for class_name, prob in result['probabilities'].items():
                # Create a color-coded progress bar
                color = {
                    'Recyclable': '#2196f3',
                    'Compostable': '#4caf50',
                    'Landfill': '#f44336'
                }[class_name]
                
                st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: 10px 0;">
                        <div style="width: 100px;">{class_name}</div>
                        <div style="flex-grow: 1; background-color: #e0e0e0; height: 20px; border-radius: 10px; margin: 0 10px;">
                            <div style="width: {prob}%; background-color: {color}; height: 100%; border-radius: 10px;"></div>
                        </div>
                        <div style="width: 60px; text-align: right;">{prob:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

# Add some information about the model
with st.expander("About this Model"):
    st.markdown("""
        This waste classification model was trained on a diverse dataset of waste items to help identify whether items are:
        
        - **Recyclable**: Items that can be processed and reused
        - **Compostable**: Organic materials that can decompose naturally
        - **Landfill**: Items that should be disposed of in regular waste
        
        The model uses a CNN architecture and has been trained to recognize various types of waste items with high accuracy.
    """) 