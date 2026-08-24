import unittest
import os
from src.utils import get_config

class TestConfig(unittest.TestCase):
    def test_config_loading(self):
        config = get_config()
        self.assertIn("threshold", config)
        self.assertIsInstance(config["threshold"], float)
        self.assertIn("re_encode", config)

if __name__ == '__main__':
    unittest.main()
