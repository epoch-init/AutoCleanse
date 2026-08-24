import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # High-quality, tiny NSFW model
        self.model_name = "FalconsAI/nsfw_image_detection"

    def load_model(self):
        """Loads the model into VRAM."""
        print(f"Loading NSFW model: {self.model_name} onto {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str) -> float:
        """
        Predicts the probability of the image being NSFW.
        The model returns two classes: 'normal' and 'nsfw'.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Model output labels: {0: 'normal', 1: 'nsfw'}
            # We apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            nsfw_score = probs[0][1].item()
            
        return nsfw_score

    def get_label(self) -> str:
        return "nudity"
