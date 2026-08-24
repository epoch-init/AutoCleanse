import unittest
import sys
import os

# Add the src directory to the path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import get_config

class TestConfig(unittest.TestCase):
    def test_config_loading(self):
        config = get_config()
        self.assertIn("threshold", config)
        self.assertIsInstance(config["threshold"], float)
        self.assertIn("re_encode", config)

if __name__ == '__main__':
    unittest.main()
