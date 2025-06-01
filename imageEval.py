import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
from faceExtractor import FaceExtractor
import cv2
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import os 
# Enable CUDA optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class Densenet:
    def __init__(self, checkpoint_path='model2.pth', batch_size=32):
        """
        Initialize the DenseNet model with optimized settings.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = self._initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scaler = GradScaler()  # For mixed precision training
        self.checkpoint_path = checkpoint_path
        self.epoch, self.metrics = self.load_model(self.checkpoint_path)
        self.model.eval()
        
        # Optimized transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Pre-compile model for faster inference (TorchScript)
        self.model = torch.jit.script(self.model)
    
    def _initialize_model(self):
        """
        Initialize an optimized DenseNet model.
        """
        model = models.densenet169(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2)
        )
        return model.to(self.device)

    def load_model(self, checkpoint_path):
        """
        Load model with optimized settings.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

    @torch.no_grad()
    def predict_batch(self, images):
        """
        Predict on a batch of images using mixed precision.
        """
        batch_tensor = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        with autocast():
            outputs = self.model(batch_tensor)
            _, predictions = torch.max(outputs, 1)
            
        
        return ['Real' if pred == 0 else 'Deepfake' for pred in predictions.cpu().numpy()]

    def predict_image(self, image):
        """
        Optimized single image prediction.
        """
        return self.predict_batch([image])[0]

    def final_imageOnly(self, image):
        """
        Wrapper method maintaining compatibility.
        """
        return self.predict_image(image)


class LimeFaceExplainer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((356, 356)),
            transforms.ToTensor()
        ])
        # Pre-compile model
        self.model = torch.jit.script(self.model)

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = checkpoint['model'].to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    @torch.no_grad()
    def _batch_predict(self, images):
        """
        Optimized batch prediction with mixed precision.
        """
        batch = torch.stack([
            self.preprocess_transform(Image.fromarray(img)) for img in images
        ]).to(self.device)
        
        with autocast():
            logits = self.model(batch)
            probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()

    def explain_face(self, face_image):
        """
        Generate LIME explanation with optimized settings.
        """
        explainer = lime_image.LimeImageExplainer(feature_selection='auto')
        explanation = explainer.explain_instance(
            np.array(face_image),
            lambda x: self._batch_predict(x),
            top_labels=1,
            hide_color=0,
            num_samples=1000,
            batch_size=32  # Increased batch size for faster processing
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        return mark_boundaries(temp / 255.0, mask)

    def visualize_explanation(self, face_image, explanation, save_path="lime_explanation.png"):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(face_image)
        plt.title("Original Face Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(explanation)
        plt.title("LIME Explanation")
        plt.axis("off")

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


def process_face(face_data, densenet, lime_explainer):
    """
    Process a single face with both detection and explanation.
    """
    face = face_data['face']
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    
    result = densenet.final_imageOnly(face_pil)
    explanation = lime_explainer.explain_face(face)
    
    return {
        'face': face,
        'result': result,
        'explanation': explanation
    }
def display_annotated_image(image, detected_faces):
    """
    Annotates the original image with bounding boxes around detected faces.

    Args:
        image (PIL.Image): The original image as a PIL Image object.
        detected_faces (list): List of detected face data, where each item contains
                               bounding box and confidence.
    """
    # Convert PIL Image to a NumPy array (ensure RGB format for OpenCV compatibility)
    image_array = np.array(image)

    # Ensure the image is in BGR format for OpenCV operations
    annotated_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    for face_data in detected_faces:
        bbox = face_data["bbox"]
        confidence = face_data["confidence"]

        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Add confidence text
        cv2.putText(
            annotated_image,
            f"{confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    # Convert back to RGB format for displaying in Streamlit
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image, caption="Annotated Image with Bounding Boxes", use_column_width=True)


def display_faces_reel(detected_faces, reel_size=5):
    """
    Displays detected faces in a reel-style format with horizontal scrolling.

    Args:
        detected_faces (list): List of detected face data, where each item is a dict
                               containing 'face' (image) and other metadata.
        reel_size (int): Number of faces to display in each reel segment.
    """
    if not detected_faces:
        st.warning("No faces to display.")
        return

    total_faces = len(detected_faces)

    # Handle cases where the number of faces is less than or equal to the reel size
    if total_faces <= reel_size:
        columns = st.columns(total_faces)
        for idx, face_data in enumerate(detected_faces):
            with columns[idx]:
                st.image(face_data["face"], use_column_width=True, clamp=True, caption=f"Face {idx + 1}")
        return

    # Add a slider to scroll through the faces
    reel_index = st.slider(
        "Scroll through detected faces:",
        min_value=0,
        max_value=total_faces - reel_size,
        step=1,
        value=0,
    )

    # Get the faces to display in the current reel
    reel_faces = detected_faces[reel_index : reel_index + reel_size]

    # Create columns to display faces side by side
    columns = st.columns(len(reel_faces))
    for idx, face_data in enumerate(reel_faces):
        with columns[idx]:
            st.image(face_data["face"], use_column_width=True, clamp=True, caption=f"Face {reel_index + idx + 1}")
def detect_deepfakes(image):
    

    # Initialize models
    model_path = 'C:/languages/deepfake-detection-1/models/FFPP_model_20_epochs_99acc.pt'
    lime_explainer = LimeFaceExplainer(model_path)
    densenet = Densenet(checkpoint_path='C:/languages/streamlit/pages/model2.pth', batch_size=32)

    # Extract faces from the uploaded image
    face_detector = FaceExtractor()
    faces = face_detector.extract_faces(image)
    display_faces_reel(faces, len(faces))
    display_annotated_image(image, faces)

    # Show loading icon while processing faces
    with st.spinner('Detecting deepfakes...'):
        # Process faces in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_face = {
                executor.submit(
                    process_face,
                    face_data,
                    densenet,
                    lime_explainer
                ): idx for idx, face_data in enumerate(faces)
            }

            # Collect results
            results = []
            for future in as_completed(future_to_face):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    st.error(f"Face processing failed: {str(e)}")

    # Generate detailed report
    st.write("The following faces were detected in the image:")

    # Display real and deepfake faces separately
    real_faces_cols = st.columns(4)
    deepfake_faces_cols = st.columns(4)
    real_faces_count = 0
    deepfake_count = 0

    for i, result in enumerate(results):
        face_num = future_to_face[list(future_to_face.keys())[i]]
        if result['result'] == 'Deepfake':
            with deepfake_faces_cols[deepfake_count % 4]:
                st.write(f"Face {face_num}: Detected as a Deepfake")
                st.image(result['face'], caption=f"Face {face_num}", use_column_width=True, clamp=True)

                # Add a button for LIME processing
                if (True):
                    st.markdown(f"**LIME Explanation:**")
                    lime_explanation_path = f"lime_explanation_{face_num}.png"
                    lime_explainer.visualize_explanation(
                        result['face'],
                        result['explanation'],
                        lime_explanation_path
                    )
                    st.image(lime_explanation_path, caption=f"LIME Explanation for Face {face_num}", use_column_width=True, clamp=True)
            deepfake_count += 1
        else:
            with real_faces_cols[real_faces_count % 4]:
                st.write(f"Face {face_num}: Detected as Real")
                st.image(result['face'], caption=f"Face {face_num}", use_column_width=True, clamp=True)
            real_faces_count += 1

    # Display summary statistics
    st.markdown("## Detection Summary")
    total_faces = len(results)
    deepfake_count = sum(1 for result in results if result['result'] == 'Deepfake')
    real_count = total_faces - deepfake_count

    # Use Streamlit columns for a neat layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Faces Detected", value=total_faces)

    with col2:
        st.metric(label="Deepfake Faces", value=deepfake_count)

    with col3:
        st.metric(label="Real Faces", value=real_count)

    st.divider()  # Adds a horizontal line for separation

def detect_deepfakes_frames(images):
   

    # Initialize models
    model_path = 'C:/languages/deepfake-detection-1/models/FFPP_model_20_epochs_99acc.pt'
    lime_explainer = LimeFaceExplainer(model_path)
    densenet = Densenet(checkpoint_path='C:/languages/streamlit/pages/model2.pth', batch_size=32)

    # Extract faces from the uploaded images
    face_detector = FaceExtractor()
    faces = []
    for image in images:
        faces.extend(face_detector.extract_faces(image,scale_factor=1.7))
    display_faces_reel(faces, len(faces))
    if (len(faces)>10):
        st.write("---------")
        faces=faces[0:10]
    # Show loading icon while processing faces
    with st.spinner('Detecting deepfakes...'):
        # Process faces in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_face = {
                executor.submit(
                    process_face,
                    face_data,
                    densenet,
                    lime_explainer
                ): idx for idx, face_data in enumerate(faces)
            }

            # Collect results
            results = []
            for future in as_completed(future_to_face):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    st.error(f"Face processing failed: {str(e)}")

    # Generate detailed report
    st.write("The following faces were detected in the image:")

    # Display real and deepfake faces separately
    real_faces_cols = st.columns(4)
    deepfake_faces_cols = st.columns(4)
    real_faces_count = 0
    deepfake_count = 0

    for i, result in enumerate(results):
        face_num = future_to_face[list(future_to_face.keys())[i]]
        if result['result'] == 'Deepfake':
            with deepfake_faces_cols[deepfake_count % 4]:
                st.write(f"Face {face_num}: Detected as a Deepfake")
                st.image(result['face'], caption=f"Face {face_num}", use_column_width=True, clamp=True)

                # Add a button for LIME processing
                if (True):
                    st.markdown(f"**LIME Explanation:**")
                    lime_explanation_path = f"lime_explanation_{face_num}.png"
                    lime_explainer.visualize_explanation(
                        result['face'],
                        result['explanation'],
                        lime_explanation_path
                    )
                    st.image(lime_explanation_path, caption=f"LIME Explanation for Face {face_num}", use_column_width=True, clamp=True)
            deepfake_count += 1
        else:
            with real_faces_cols[real_faces_count % 4]:
                st.write(f"Face {face_num}: Detected as Real")
                st.image(result['face'], caption=f"Face {face_num}", use_column_width=True, clamp=True)
            real_faces_count += 1

    # Display summary statistics
    st.markdown("## Detection Summary")
    total_faces = len(results)
    deepfake_count = sum(1 for result in results if result['result'] == 'Deepfake')
    real_count = total_faces - deepfake_count

    # Use Streamlit columns for a neat layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Faces Detected", value=total_faces)

    with col2:
        st.metric(label="Deepfake Faces", value=deepfake_count)

    with col3:
        st.metric(label="Real Faces", value=real_count)

    st.divider()
def api_detect_deepfakes(image):
   

    # Initialize models
    model_path = 'C:/languages/deepfake-detection-1/models/FFPP_model_20_epochs_99acc.pt'
    
    densenet = Densenet(checkpoint_path='C:/languages/streamlit/pages/model2.pth', batch_size=32)

    # Extract faces from the uploaded image
    face_detector = FaceExtractor()
    faces = face_detector.extract_faces(image)

    for i in faces :
        result=densenet.final_imageOnly(i)
        if result=="Deepfake":
            return
    return result 
    

    # Show loading icon while processing faces
    
       
    
