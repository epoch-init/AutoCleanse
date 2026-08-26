import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.35): 
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../weights/clip-vit")
        self.model_name = "openai/clip-vit-base-patch32"
        
        # Refined Labels to solve False Positives (Skin vs Nudity)
        self.neutral_labels = [
            "a photo of people wearing clothes",
            "a close up of a human face",
            "a photo of people talking",
            "a person's arms and hands"
        ]
        self.nsfw_labels = [
            "a photo of explicit nudity", 
            "exposed female breasts",
            "uncovered genitalia"
        ]
        self.all_labels = self.neutral_labels + self.nsfw_labels

    def load_model(self):
        print(f"Loading Optimized CLIP (FP16: {torch.cuda.is_available()})...")
        load_path = self.model_path if os.path.exists(self.model_path) else self.model_name
        
        self.processor = CLIPProcessor.from_pretrained(load_path)
        self.model = CLIPModel.from_pretrained(load_path)
        
        # Speed hack: Use Half Precision (FP16) if on GPU
        if torch.cuda.is_available():
            self.model = self.model.half()
            
        self.model.to(self.device)
        self.model.eval()

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
            self.processor.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)

    def predict_batch(self, pil_images) -> list:
        """
        Processes a list of images at once for 3x-5x more speed.
        """
        inputs = self.processor(
            text=self.all_labels, 
            images=pil_images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        # Ensure inputs are half precision if model is
        if torch.cuda.is_available():
            inputs['pixel_values'] = inputs['pixel_values'].half()

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Softmax across the labels for each image in the batch
            probs = outputs.logits_per_image.softmax(dim=1)
            
            batch_scores = []
            for i in range(len(pil_images)):
                # NSFW score = sum of NSFW probabilities
                nsfw_prob = sum(probs[i][j].item() for j in range(len(self.neutral_labels), len(self.all_labels)))
                neutral_prob = sum(probs[i][j].item() for j in range(len(self.neutral_labels)))
                
                # Logic Fix: If Neutral is clearly dominant, suppress the NSFW score
                if neutral_prob > nsfw_prob:
                    batch_scores.append(nsfw_prob * 0.4)
                else:
                    batch_scores.append(nsfw_prob)
                    
            return batch_scores

    def predict(self, image_input):
        # Fallback for single images
        return self.predict_batch([image_input])[0]

    def get_label(self) -> str:
        return "nudity"
