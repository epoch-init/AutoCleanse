import os
from nudenet import NudeDetector
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.detector = None
        # Labels that trigger a cut (includes partial and full nudity)
        self.target_labels = [
            "EXPOSED_BREAST_F", 
            "EXPOSED_GENITALIA_F", 
            "EXPOSED_BUTTOCKS", 
            "EXPOSED_ANUS",
            "EXPOSED_GENITALIA_M"
        ]

    def load_model(self):
        """
        Initializes the NudeDetector. 
        NudeNet handles weights automatically but we can force it 
        to be offline by ensuring it's loaded.
        """
        print("Loading NudeNet Detector (YOLOv8-based specialized weights)...")
        # By default, NudeDetector looks in ~/.NudeNet/
        # This will run on GPU if onnxruntime-gpu is installed.
        self.detector = NudeDetector()

    def predict(self, image_path: str) -> float:
        """
        Detects body parts. If any explicit part is found, 
        returns the highest confidence score.
        """
        # Returns a list of dicts: [{'class': 'LABEL', 'score': 0.9, 'box': [...]}, ...]
        detections = self.detector.detect(image_path)
        
        max_score = 0.0
        for det in detections:
            if det['class'] in self.target_labels:
                if det['score'] > max_score:
                    max_score = det['score']
        
        return max_score

    def get_label(self) -> str:
        return "nudity"
