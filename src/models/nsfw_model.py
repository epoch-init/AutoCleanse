import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.15): 
        # Lower threshold here because we are now using relative "NSFW vs Neutral" logic
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../weights/clip-vit")
        self.model_name = "openai/clip-vit-base-patch32"
        
        # ANCHOR LABELS: Adding more neutral labels prevents false positives
        self.neutral_labels = [
            "a photo of a person wearing clothes",
            "a photo of a person's face",
            "a photo of a person wearing a shirt",
            "a photo of a person in a room"
        ]
        self.nsfw_labels = [
            "a photo of explicit nudity", 
            "a photo of exposed breasts",
            "a photo of genitalia"
        ]
        self.all_labels = self.neutral_labels + self.nsfw_labels

    def load_model(self):
        print(f"Loading CLIP-ViT with Balanced Prompts...")
        load_path = self.model_path if os.path.exists(self.model_path) else self.model_name
        self.processor = CLIPProcessor.from_pretrained(load_path)
        self.model = CLIPModel.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
            self.processor.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)

    def predict(self, image_input) -> float:
        image = image_input if not isinstance(image_input, str) else Image.open(image_input).convert("RGB")
        
        inputs = self.processor(
            text=self.all_labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Logits represent the raw similarity
            probs = outputs.logits_per_image.softmax(dim=1)
            
            # Sum probability of all NSFW categories
            # Neutral are 0-3, NSFW are 4-6
            nsfw_prob = sum(probs[0][i].item() for i in range(len(self.neutral_labels), len(self.all_labels)))
            
            # Sum probability of all Neutral categories
            neutral_prob = sum(probs[0][i].item() for i in range(len(self.neutral_labels)))

            # Relative Score: How much more likely is it to be NSFW than Neutral?
            # If neutral is much higher, we return 0.
            if neutral_prob > nsfw_prob:
                return nsfw_prob * 0.5 # Penalize the score if neutral dominates
            
            return nsfw_prob

    def get_label(self) -> str:
        return "nudity"
