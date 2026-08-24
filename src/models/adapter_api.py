from typing import List
from .base import BaseModelAdapter

class ModelInterface:
    """
    Unified API that communicates with one or multiple model adapters.
    """
    def __init__(self, adapters: List[BaseModelAdapter]):
        self.adapters = adapters

    def analyze_frame(self, image_path: str):
        """
        Runs the frame through all registered adapters.
        Returns a dictionary of labels and their scores.
        """
        results = {}
        for adapter in self.adapters:
            score = adapter.predict(image_path)
            results[adapter.get_label()] = score
        return results
