# tests/test_nudenet.py
import unittest
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.nsfw_model import NsfwModelAdapter

class TestNudeNet(unittest.TestCase):
    def test_detection(self):
        adapter = NsfwModelAdapter()
        adapter.load_model()
        
        # Create a blank image
        img_path = "blank.jpg"
        Image.new('RGB', (640, 640), color='white').save(img_path)
        
        score = adapter.predict(img_path)
        self.assertEqual(score, 0.0) # White square should have 0 nudity
        os.remove(img_path)

if __name__ == '__main__':
    unittest.main()
