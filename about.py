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
        st.title("About Us")

        # Subtitle - Project Information
        st.markdown("## Deception: Advances in Deepfake Detection")

        # Project Description
        st.write("""
        Our project, Deception: Advances in Deepfake Detection, focuses on improving the detection of deepfake content in *Audio, **Video, and **Text* formats. 
        The goal is to build robust solutions to identify and mitigate the impact of deepfakes, ensuring the authenticity of digital media.
        """)

        # Team Section Title
        st.header("Our Team: DEADLOCK")

        # Team Member Info
        team_members = [
            {
                "name": "Chandrahaas Jasti",
                
                "email": "chandrahaasjasti@gmail.com",
                "image": "image.png"  # Replace with actual image path
            },
            {
                "name": "Nitish Narva",
                
                "email": "nitishnarvalpt@gmail.com",
                "image": "image.png"  # Replace with actual image path
            },
            {
                "name": "Ananya Sirandass",
                
                "email": "ananyasirandass@gmail.com",
                "image": "image.png"  # Replace with actual image path
            },
            {
                "name": "Sree Vibha Panchagnula",
                
                "email": "vibhasree2811@gmail.com",
                "image": "image.png"  # Replace with actual image path
            },
            {
                "name": "Sathvik Thatipally",
               
                "email": "sathvikt123@gmail.com",
                "image": "image.png"  # Replace with actual image path
            }
        ]

        # Layout to display team member cards (five members per row)
        for i in range(0, len(team_members), 5):
            cols = st.columns(5)
            for col, member in zip(cols, team_members[i:i + 5]):
                with col:
                    # Load member image using the load_image function
                    image = load_image(member["image"])
                    st.image(image, width=150)
                    st.markdown(f"{member['name']}")
                   
                    st.write(f"[{member['email']}]({member['email']})")
                    st.write("----")

        # Footer or additional information
        st.write("""
        We strive to deliver cutting-edge deepfake detection solutions by working collaboratively to address the growing threats posed by synthetic media.
        """)

