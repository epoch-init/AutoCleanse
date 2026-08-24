from abc import ABC, abstractmethod

class BaseModelAdapter(ABC):
    """
    Abstract Base Class for all Model Adapters.
    Ensures that any model used by AutoCleanse follows the same interface.
    """
    
    @abstractmethod
    def load_model(self):
        """Initialize and load the model into VRAM."""
        pass

    @abstractmethod
    def predict(self, image_path: str) -> float:
        """
        Takes a path to an image and returns a probability score (0.0 to 1.0).
        """
        pass

    @abstractmethod
    def get_label(self) -> str:
        """Returns the type of content this model detects (e.g., 'nudity')."""
        pass
