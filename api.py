from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import imageEval as E
app = FastAPI()

class ImageData(BaseModel):
    base64_image: str

@app.post("/process_image/")
async def process_image(image_data: ImageData):
    try:
        # Decode the Base64 image
        image_bytes = base64.b64decode(image_data.base64_image)
        # Convert to NumPy array
        np_array = np.frombuffer(image_bytes, np.uint8)
        # Decode image using OpenCV
        image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise ValueError("The provided data is not a valid image.")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result=E.api_detect_deepfakes(image_rgb)
        
        # Placeholder logic for determining "fake" or "real"
        # You should replace this with your deepfake detection model's logic
        # For now, we assume if the width > 1000, it's "real", otherwise "fake"
       # height, width, _ = image_rgb.shape
        #result = "real" if width > 1000 else "fake"

        return {"message": result}

    except (base64.binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
