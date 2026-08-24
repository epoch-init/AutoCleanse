import torch
from ultralytics import YOLO
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.35):
        # We lower the default threshold for object detection because 
        # even a low-confidence detection of a nipple should trigger a cut.
        self.threshold = threshold
        self.model = None
        # This is a specialized YOLOv8 model for NSFW detection
        # It detects: 'exposed_breast', 'exposed_buttocks', 'exposed_genitalia', etc.
        self.model_path = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt" 
        # Note: In a production "AutoCleanse", we would use weights specifically 
        # fine-tuned for NSFW. For this implementation, we'll use the detection logic 
        # that looks for specific 'exposed' classes.
        self.target_classes = ["exposed_breast", "exposed_buttocks", "exposed_genitalia", "covered_breast"]

    def load_model(self):
        print(f"Loading YOLOv8 NSFW Detector...")
        # Using the Nano (n) version for maximum speed and lowest VRAM (approx 0.5GB)
        # We use a model specifically trained on the 'NSFW' dataset
        # For this example, we'll load a model from a reliable source
        self.model = YOLO('yolov8n.pt') # Placeholder: Replace with 'weights/nsfw_yolov8.pt'
        
        if torch.cuda.is_available():
            self.model.to('cuda')

    def predict(self, image_path: str) -> float:
        """
        Runs object detection on the frame.
        If any forbidden body parts are detected above the threshold, returns the max confidence.
        """
        results = self.model(image_path, verbose=False)[0]
        
        max_score = 0.0
        
        # results.boxes contains the detected objects
        for box in results.boxes:
            # Get the class name
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            confidence = float(box.conf[0])
            
            # If the detected object is in our 'explicit' list
            # For a dedicated NSFW YOLO model, classes would be like 'exposed_breast'
            if label in self.target_classes or "exposed" in label:
                if confidence > max_score:
                    max_score = confidence
        
        return max_score

    def get_label(self) -> str:
        return "nudity"
