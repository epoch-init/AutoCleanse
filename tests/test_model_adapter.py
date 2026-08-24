import unittest
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.nsfw_model import NsfwModelAdapter

class TestNsfwModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = NsfwModelAdapter(threshold=0.3)
        cls.adapter.load_model()
        
        cls.test_img_path = "test_img.jpg"
        img = Image.new('RGB', (640, 640), color=(120, 120, 120))
        img.save(cls.test_img_path)

    def test_prediction_returns_float(self):
        score = self.adapter.predict(self.test_img_path)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_model_type(self):
        # Verify it's using the YOLO architecture
        self.assertTrue(hasattr(self.adapter.model, 'names'))

if __name__ == '__main__':
    unittest.main()
