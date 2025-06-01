import streamlit as st
from streamlit_option_menu import option_menu
import Textsum as ts
# Sample function for chatbot response
maindata=""
def get_chatbot_response(user_input):
    # Placeholder for actual chatbot logic
    return f"You said: {user_input}"

def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "DeepFake Detection", "Take a Photo", "API"],
        icons=["house", "camera",  "photo", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0!important",
                "width": "100%",
                "max-width": "1200px",
                "margin": "0 auto",
            }
        }
    )
    return selected

def main():
    selected = streamlit_menu()
    st.markdown(
        """
        <style>
        .reportview-container {
            max-width: 130000px;  /* Adjust this value as needed */
            margin: 0 auto;      /* Center the content */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create the sidebar for chatbot interface
    with st.sidebar:
        st.header("Chatbot")
        
        user_input = st.text_input("Ask me anything:")
        #user_input+=maindata
        
        if user_input:
            with st.spinner('Processing your request...'):
                response = ts.all_in_one.askgem(user_input)
                st.text_area("Chatbot Response:", value=response, height=600)

    # Dictionary to map menu options to functions
    pages = {
        "Home": home_page,
        "DeepFake Detection": deepfake_detection_page,
        
        "Take a Photo": take_photo_page,
        "API": about_page
    }

    # Execute the corresponding function based on the selected option
    if selected in pages:
        pages[selected]()  # Call the respective function

def home_page():
    st.markdown(
        """
        <style>
        .reportview-container {
            max-width: 1300px;  /* Adjust this value as needed */
            margin: 0 auto;      /* Center the content */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image("image.png", caption="Description of the image", use_column_width=True)
    st.write("In the digital age, deepfakes have emerged as a significant threat to the authenticity of online content. These sophisticated AI-generated videos can convincingly mimic real people, making it increasingly difficult to distinguish fact from fiction. However, as the technology behind deepfakes has advanced, so too have the tools and techniques designed to detect them. In this blog, we will explore the top five deepfake detection tools and techniques available today.We will try our best to spread and detect misinformation")

    st.title("Deepfake Awareness:")
    st.subheader("Concise ways humans can detect deepfakes:")
    st.write("""Unnatural Facial Expressions: Look for forced or mismatched emotions.\nMismatched Lip Sync: Check if lip movements align with spoken words.\nEye Movement: Watch for static or unnatural eye behavior. \nBackground Artifacts: Notice any distortion or inconsistencies in the background.\n Jerky Body Movements: Observe if body language appears robotic or unnatural.""")
    # First row for displaying YouTube frames
    st.subheader("YouTube Videos")
    
    # Create three columns for YouTube videos
    col1, col2, col3 = st.columns(3)
    
    # Add YouTube video frames (example video links)
    with col1:
        st.video("https://www.youtube.com/watch?v=pkF3m5wVUYI")  # Replace with your video URL
    with col2:
        st.video("https://www.youtube.com/watch?v=UcQet3Tcx9M")  # Replace with your video URL
    with col3:
        st.video("https://www.youtube.com/watch?v=S951cdansBI")  # Replace with your video URL

    # Second row for displaying news articles
    st.subheader("Latest News Articles")
    
    # You can add articles as links or any other format you prefer
    news_articles = [
        {
            "title": "Understanding Deepfakes: What You Need to Know",
            "url": "https://example.com/article1"
        },
        {
            "title": "The Rise of Deepfake Technology and Its Impact",
            "url": "https://example.com/article2"
        },
        {
            "title": "Deepfake Detection Techniques and Tools",
            "url": "https://example.com/article3"
        },
    ]

    for article in news_articles:
        st.markdown(f"- [{article['title']}]({article['url']})")

def deepfake_detection_page():
    import deepfake_detection
    deepfake_detection.run()





def take_photo_page():
    import take_photo
    #hello
    take_photo.run()
  

def about_page():
    import about
    about.run()

if __name__ == "__main__":
    main()
