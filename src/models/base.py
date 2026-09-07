from abc import ABC, abstractmethod
from typing import List, Any

class BaseModelAdapter(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image_input: Any) -> float:
        pass
        
    @abstractmethod
    def predict_batch(self, images: List[Any]) -> List[float]:
        """Processes a list of images simultaneously for massive speedups."""
        pass

    @abstractmethod
    def get_label(self) -> str:
        pass
