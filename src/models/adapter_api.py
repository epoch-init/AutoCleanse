from typing import List
from .base import BaseModelAdapter

class ModelInterface:
    def __init__(self, adapters: List[BaseModelAdapter]):
        self.adapters = adapters

    def analyze_frame(self, image_input):
        results = {}
        for adapter in self.adapters:
            results[adapter.get_label()] = adapter.predict(image_input)
        return results

    def analyze_batch(self, images: List):
        """Returns a list of dictionaries (one for each image in the batch)"""
        batch_results = [{} for _ in range(len(images))]
        
        for adapter in self.adapters:
            scores = adapter.predict_batch(images)
            for i, score in enumerate(scores):
                batch_results[i][adapter.get_label()] = score
                
        return batch_results
