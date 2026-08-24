import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.25): 
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Local paths for offline use
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../weights/clip-vit")
        self.model_name = "openai/clip-vit-base-patch32"
        
        self.text_inputs = None
        # These specific prompts help CLIP identify cinematic nudity vs clothing
        self.labels = [
            "a photo of people wearing clothes", 
            "a photo of explicit nudity", 
            "a photo of exposed breasts"
        ]

    def load_model(self):
        print(f"Loading CLIP-ViT Model (Device: {self.device})...")
        
        load_path = self.model_path if os.path.exists(self.model_path) else self.model_name
        
        self.processor = CLIPProcessor.from_pretrained(load_path)
        self.model = CLIPModel.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()

        if not os.path.exists(self.model_path):
            print("Downloading and saving weights for offline use...")
            os.makedirs(self.model_path, exist_ok=True)
            self.processor.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)

        # Pre-tokenize text prompts
        self.text_inputs = self.processor(
            text=self.labels, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

    def predict(self, image_input) -> float:
        """
        Accepts a PIL Image. Uses CLIP's internal logit scaling for high accuracy.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input

        # Prepare inputs for the model
        inputs = self.processor(
            text=self.labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            # Standard CLIP forward pass - most robust way to get logits
            outputs = self.model(**inputs)
            
            # logits_per_image is the similarity score between the image and the 3 labels
            logits_per_image = outputs.logits_per_image 
            
            # Convert to probabilities using softmax
            probs = logits_per_image.softmax(dim=1)
            
            # probs[0][0] is 'clothed'
            # probs[0][1] is 'nudity'
            # probs[0][2] is 'exposed breasts'
            # We sum the nudity-related probabilities
            nsfw_score = probs[0][1].item() + probs[0][2].item()
            
        return nsfw_score

    def get_label(self) -> str:
        return "nudity"
