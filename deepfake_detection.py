import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from datetime import datetime
import logging
import streamlit as st
import cv2
import tempfile
import shutil
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import joblib
import imageEval as E

data = ""

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def create_temp_file(uploaded_file):
    """Create a temporary file from uploaded content"""
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        logging.error(f"Error creating temporary file: {str(e)}")
        raise

def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f"Error deleting temporary file {file_path}: {str(e)}")

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    try:
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Save audio to temporary file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_path = temp_audio.name
        temp_audio.close()
        
        audio.write_audiofile(audio_path, verbose=False, logger=None)
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from video: {str(e)}")
        raise
    finally:
        if 'video' in locals():
            video.close()

def extract_features(file_path, n_mfcc=13):
    """Extract MFCC features from audio"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logging.error(f"Error extracting audio features: {str(e)}")
        raise

def predict_audio_deepfake(audio_path, model):
    """Predict if audio is deepfake"""
    try:
        features = extract_features(audio_path)
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        logging.error(f"Error predicting audio deepfake: {str(e)}")
        raise

def extract_frames(video_path, desired_fps=1):
    """Extract frames from video at specified FPS"""
    frames = []
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError("Unable to open video file")

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Invalid FPS value detected")

        frame_interval = int(fps / desired_fps)
        frame_count = 0

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                # Convert the frame to RGB color space
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1

        return frames, len(frames)
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return [], 0

def analyze_frame(frame):
    """Analyze a single frame for deepfake detection"""
    try:
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = E.Densenet.final_imageOnly(frame_image)
        return result, frame_image
    except Exception as e:
        logging.error(f"Error analyzing frame: {str(e)}")
        raise

def run():
    """Main application function"""
    setup_logging()
    temp_files = []
 
    try:
        st.title("DeepFake Detection")
        st.write("Upload your media file (video, image, or audio).")

        # Load the audio detection model
        try:
            audio_model = joblib.load('knn_model2.pkl')
        except Exception as e:
            st.error(f"Error loading audio detection model: {str(e)}")
            st.write("Audio detection will be disabled.")
            audio_model = None

        media_file = st.file_uploader(
            "Upload Media",
            type=["mp4", "avi", "mov", "jpg", "jpeg", "png", "mp3", "wav", "ogg"]
        )

        if media_file is None:
            return

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process the uploaded file
        temp_input_path = create_temp_file(media_file)
        temp_files.append(temp_input_path)
        
        file_extension = media_file.name.split('.')[-1].lower()

        if file_extension in ["mp4", "avi", "mov"]:
            status_text.text("Processing video...")

            # Extract and analyze frames
            frames, frame_count = extract_frames(temp_input_path)
            st.write(f"Analyzing {frame_count} frames...")

            E.detect_deepfakes_frames(frames)
            
            # Process audio if model is available
            if audio_model:
                status_text.text("Processing audio...")
                try:
                    audio_path = extract_audio_from_video(temp_input_path)
                    temp_files.append(audio_path)
                    
                    # Display audio player
                    st.audio(audio_path)
                    
                    # Analyze audio
                    result = predict_audio_deepfake(audio_path, audio_model)
                    if result == 0:
                        st.error("⚠ DEEPFAKE AUDIO DETECTED!")
                        st.write("The audio appears to be artificially generated.")
                    else:
                        st.success("✅ AUTHENTIC AUDIO")
                        st.write("This audio appears to be genuine.")
                except Exception as e:
                    st.warning(f"Audio analysis failed: {str(e)}")
                    st.write("Continuing with video analysis only.")

            # Display overall analysis
            #deepfake_frames = frame_results.count("Deepfake")
            #if deepfake_frames > 0:
                #st.warning(f"⚠ {deepfake_frames} frames identified as potentially manipulated")

        elif file_extension in ["jpg", "jpeg", "png"]:
            status_text.text("Processing image...")
            image = Image.open(media_file).convert("RGB")
            #st.image(image, caption='Uploaded Image', use_column_width=True)
            
            result = E.detect_deepfakes(image)
            st.write(f"Image analysis result: {result}")
            progress_bar.progress(1.0)

        elif file_extension in ["mp3", "wav", "ogg"]:
            status_text.text("Processing audio...")
            if not audio_model:
                st.error("Audio detection model not available.")
                return

            try:
                # Display audio player
                st.audio(media_file)
                
                # Analyze audio
                result = predict_audio_deepfake(temp_input_path, audio_model)
                progress_bar.progress(1.0)
                
                if result == 0:
                    st.error("⚠ DEEPFAKE DETECTED!")
                    st.write("The audio appears to be artificially generated.")
                else:
                    st.success("✅ AUTHENTIC AUDIO")
                    st.write("This audio appears to be genuine.")

                # Add confidence disclaimer
                st.write("---")
                st.write("""*Note:* This detection is based on machine learning analysis and may not be 100% accurate. 
                            For critical applications, please verify results through additional means.""")
            except Exception as e:
                st.error(f"Audio analysis failed: {str(e)}")
                return

        status_text.text("Processing complete!")
        progress_bar.progress(1.0)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Application error: {str(e)}")
    finally:
        cleanup_temp_files(temp_files)


