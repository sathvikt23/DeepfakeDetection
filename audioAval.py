import streamlit as st
import librosa
import numpy as np
import joblib
from moviepy.editor import VideoFileClip  # Correct import
import tempfile
import os

def extract_audio_from_video(video_file):
    """
    Extract audio from uploaded video file
    """
    # Create a temporary file to save the video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        video_path = tmpfile.name
    
    try:
        # Extract audio using moviepy
        video = VideoFileClip(video_path)  # Use VideoFileClip
        audio = video.audio
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio_tmpfile:
            audio_path = audio_tmpfile.name
            audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        return audio_path
    
    finally:
        # Clean up temporary video file
        os.unlink(video_path)

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)
    
def predict_deepfake(audio_path, model):
    """
    Predict if audio is deepfake using loaded model
    """
    # Extract features
    features = extract_features(audio_path)
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    return prediction[0]

def main():
    st.title("Deepfake Audio Detection")
    st.write("Upload a video or audio file to check if it contains deepfake audio")
    
    # Load the trained model
    try:
        model = joblib.load('knn_model2.pkl')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['mp4', 'wav', 'mp3']
    )
    
    if uploaded_file is not None:
        try:
            st.write("Processing file...")

            # Create progress bar
            progress_bar = st.progress(0)
            
            # Process based on file type
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type in ['mp4']:
                # Extract audio from video
                progress_bar.progress(30)
                st.write("Extracting audio from video...")
                audio_path = extract_audio_from_video(uploaded_file)
            else:
                # Save audio file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmpfile:
                    tmpfile.write(uploaded_file.read())
                    audio_path = tmpfile.name
            
            # Update progress
            progress_bar.progress(60)
            st.write("Analyzing audio...")
            
            # Make prediction
            result = predict_deepfake(audio_path, model)
            
            # Update progress
            progress_bar.progress(100)
            
            # Display result with appropriate styling
            st.write("---")
            st.write("### Result")
            if result == 1:  # Adjust based on your model's output
                st.error("⚠ DEEPFAKE DETECTED!")
                st.write("This audio appears to be artificially generated.")
            else:
                st.success("✅ AUTHENTIC AUDIO")
                st.write("This audio appears to be genuine.")
            
            # Clean up temporary file
            os.unlink(audio_path)
            
            # Add confidence disclaimer
            st.write("---")
            st.write("""*Note:* This detection is based on machine learning analysis and may not be 100% accurate. 
                        For critical applications, please verify results through additional means.""")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Add information about supported formats
    st.sidebar.write("### Supported Formats")
    st.sidebar.write("- Video: MP4")
    st.sidebar.write("- Audio: WAV, MP3")
    
    # Add information about the model
    st.sidebar.write("### About")
    st.sidebar.write("""This app uses a KNN classifier trained on MFCC features to detect deepfake audio.
                        The model analyzes the acoustic properties of the audio to determine if it's genuine or artificially generated.""")

if __name__== "__main__":  # Fix the condition to run the main function
    main()
