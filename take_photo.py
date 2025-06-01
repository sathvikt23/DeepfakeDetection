import streamlit as st
from PIL import Image
import imageEval as E
# Placeholder for your image processing class/function


# Create an instance of your image processing class


def run():
    st.title("Image Processing App")
    st.write("This section allows you to take a photo or upload an image for processing.")

    # Camera input for taking a photo
    photo = st.camera_input("Click the button below to take a photo")

    if photo is not None:
        # Process the taken photo
        status_text = st.empty()  # Placeholder for processing status
        status_text.text("Processing image...")
        
        image = Image.open(photo).convert("RGB")
        st.image(image, caption='Taken Photo', use_column_width=True)
        
        # Call the image processing function
        E.detect_deepfakes(image)
        #st.write(f"Image analysis result: {result}")

    # File uploader for image upload
    media_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if media_file is not None:
        # Process the uploaded image
        status_text = st.empty()  # Placeholder for processing status
        status_text.text("Processing image...")
        
        image = Image.open(media_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        E.detect_deepfakes(image)
        # Call the image processing function
        #result = E.detect_deepfakes(image)
        #data=""
        #data+=f"Image analysis result: {result}"
        #st.write(f"Image analysis result: {result}")
    #return f"These are the analysis of current deepfake analysis {data}\nOnly answer when asked about it ."

if __name__ == '__main__':
    run()
