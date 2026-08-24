import os
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.3): # Lower threshold for cinematic sensitivity
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Path to local model for 100% offline use
        # On first run, it will download to cache. You can then copy that cache 
        # to this path and it will never need internet again.
        self.model_path = os.path.join(os.path.dirname(__file__), "../../weights/nsfw_vit")
        self.model_name = "FalconsAI/nsfw_image_detection"

    def load_model(self):
        print(f"Loading High-Sensitivity ViT Model...")
        
        load_path = self.model_path if os.path.exists(self.model_path) else self.model_name
        
        self.processor = AutoImageProcessor.from_pretrained(load_path)
        self.model = AutoModelForImageClassification.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Save for future offline use if we just downloaded it
        if not os.path.exists(self.model_path):
            print("Saving model locally for offline use...")
            os.makedirs(self.model_path, exist_ok=True)
            self.processor.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)

    def _run_inference(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Label 1 is usually the 'nsfw' or 'unsafe' class
            return probs[0][1].item()

    def predict(self, image_path: str) -> float:
        full_img = Image.open(image_path).convert("RGB")
        
        # Scale 1: Full Frame
        score_full = self._run_inference(full_img)
        
        # Scale 2: Center Crop (Helps with small subjects in cinematic shots)
        w, h = full_img.size
        # Crop the middle 60% of the image
        left = w * 0.2
        top = h * 0.2
        right = w * 0.8
        bottom = h * 0.8
        crop_img = full_img.crop((left, top, right, bottom))
        score_crop = self._run_inference(crop_img)
        
        # Return the higher of the two scores
        return max(score_full, score_crop)

    def get_label(self) -> str:
        return "nudity"