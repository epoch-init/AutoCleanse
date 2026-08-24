import torch
import torchvision.transforms as transforms
from PIL import Image
from .base import BaseModelAdapter

class NsfwModelAdapter(BaseModelAdapter):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        # Using SqueezeNet as it's highly efficient for < 2GB VRAM
        from torchvision.models import squeezenet1_1
        self.model = squeezenet1_1(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(self.device)

        with torch.no_grad():
            out = self.model(batch_t)
            # Standardizing output to a 0.0 - 1.0 probability
            prob = torch.nn.functional.softmax(out, dim=1)
            # In a real scenario, index 1 would be the 'NSFW' class
            return prob[0][1].item() 

    def get_label(self) -> str:
        return "nudity"
