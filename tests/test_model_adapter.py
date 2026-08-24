import unittest
import sys
import os
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.nsfw_model import NsfwModelAdapter

class TestNsfwModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests in this class to save time/VRAM."""
        cls.adapter = NsfwModelAdapter(threshold=0.5)
        cls.adapter.load_model()
        
        # Create a dummy image for testing
        cls.test_img_path = "test_img.jpg"
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        img.save(cls.test_img_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.remove(cls.test_img_path)

    def test_model_loading(self):
        self.assertIsNotNone(self.adapter.model)
        self.assertEqual(str(self.adapter.device), "cuda" if torch.cuda.is_available() else "cpu")

    def test_prediction_range(self):
        score = self.adapter.predict(self.test_img_path)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_label_consistency(self):
        self.assertEqual(self.adapter.get_label(), "nudity")

if __name__ == '__main__':
    unittest.main()
