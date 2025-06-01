import streamlit as st
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image

class FaceExtractor:
    def __init__(self):
        # Initialize the MTCNN face detector
        self.detector = MTCNN()

    def extract_faces(self, image, scale_factor=1.6):
       
        image_array = np.array(image)

        # ensuring image to be in 3 channels  (convert RGBA to RGB),important check 
        if image_array.shape[-1] == 4:  # Check for alpha channel
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Detect faces in the image ,MTCNN abstraction 
        results = self.detector.detect_faces(image_array)

      
        detected_faces = []

        # Extract the faces and their bounding boxes,bounding boxes here mean  the required face grid 
        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']

            x, y, width, height = bounding_box

            # Increase the bounding box size
            x = max(0, x - int(width * (scale_factor - 1) / 2))
            y = max(0, y - int(height * (scale_factor - 1) / 2))
            width = int(width * scale_factor)
            height = int(height * scale_factor)

            face = image_array[y:y + height, x:x + width]

            # Resize the face to 224x224 pixels
            face = cv2.resize(face, (224, 224))

            detected_faces.append({
                'face': face,
                'bbox': (x, y, x + width, y + height),
                'confidence': confidence
            })

        return detected_faces