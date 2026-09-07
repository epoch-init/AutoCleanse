import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.15): 
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use FP16 for 2x speed and half VRAM usage if on GPU
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../weights/clip-vit")
        self.model_name = "openai/clip-vit-base-patch32"
        
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
        print(f"Loading CLIP-ViT in High-Speed Mode (FP16)...")
        load_path = self.model_path if os.path.exists(self.model_path) else self.model_name
        
        self.processor = CLIPProcessor.from_pretrained(load_path)
        # Load directly in Half-Precision
        self.model = CLIPModel.from_pretrained(load_path, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
            self.processor.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)

    def _calculate_score(self, probs):
        nsfw_prob = sum(probs[i].item() for i in range(len(self.neutral_labels), len(self.all_labels)))
        neutral_prob = sum(probs[i].item() for i in range(len(self.neutral_labels)))
        if neutral_prob > nsfw_prob:
            return nsfw_prob * 0.5 
        return nsfw_prob

    def predict(self, image_input) -> float:
        return self.predict_batch([image_input])[0]

    def predict_batch(self, images) -> list[float]:
        # Ensure all are PIL images
        pil_images = [img if not isinstance(img, str) else Image.open(img).convert("RGB") for img in images]
        
        inputs = self.processor(
            text=self.all_labels, 
            images=pil_images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        # Convert image tensors to match model dtype (FP16)
        if self.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            
            # Calculate scores for the whole batch
            return [self._calculate_score(p) for p in probs]

    def get_label(self) -> str:
        return "nudity"
