import streamlit as st
from PIL import Image
import os

def run():
    # Define the path for the default placeholder image
    DEFAULT_IMAGE_PATH = "image.png"  # Replace with the actual path to your placeholder image

    # Function to load images with error handling
    def load_image(image_path):
        try:
            if os.path.isfile(image_path):
                return Image.open(image_path)
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")
        except (FileNotFoundError, OSError):
            return Image.open(DEFAULT_IMAGE_PATH)  # Use the placeholder image if an error occurs

    # Title
    

    # Subtitle - Project Information
    st.markdown("## Deception: Advances in Deepfake Detection")

    # Project Description
    st.write("""
    Our project, Deception: Advances in Deepfake Detection, focuses on improving the detection of deepfake content in *Audio, **Video, and **Text* formats. 
    The goal is to build robust solutions to identify and mitigate the impact of deepfakes, ensuring the authenticity of digital media.
    """)

    # Team Section Title
    
    # Layout to display team member cards (five members per row)
    

    # Footer or additional information
    st.write("""
    We strive to deliver cutting-edge deepfake detection solutions by working collaboratively to address the growing threats posed by synthetic media.
    """)

    # Section: API Integration
    st.header("API Integration")
    st.write("Below is an example of integrating with an API to fetch deepfake detection results.")

    code_snippet = """
from torchvision import transforms
from PIL import Image 
import pickle
import json
import requests

# Image path
image_path = "image_path"

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
image = transform(image)

# Add batch dimension
tensor = image.unsqueeze(0) 

# Serialize the tensor
pickle_bytesTF = pickle.dumps(tensor)
tensor_stringTF = pickle_bytesTF.decode('latin1')

# Create JSON payload
dict = {'tensor': tensor_stringTF}
x = json.dumps(dict)

# Send the request to FastAPI(AWS)
res = requests.post("http://13.235.255.187/predict", x)
print(res.json())
"""
    st.code(code_snippet, language='python')

# Run the Streamlit app
if __name__ == "__main__":
    run()
