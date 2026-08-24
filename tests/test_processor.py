import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from processor import VideoProcessor

class TestProcessorLogic(unittest.TestCase):
    def setUp(self):
        self.config = {'padding': 1, 'min_delta_between_splits': 5}
        self.processor = VideoProcessor(self.config, None)

    def test_merge_logic(self):
        # Two scenes close together (delta 2s < min_delta 5s)
        scenes = [
            {'start': 10, 'end': 15, 'label': 'nudity'},
            {'start': 17, 'end': 20, 'label': 'nudity'}
        ]
        merged = self.processor.merge_and_pad_scenes(scenes)
        # Should be one scene: (10-1) to (20+1) = 9 to 21
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['start'], 9)
        self.assertEqual(merged[0]['end'], 21)

    def test_clean_segments_calculation(self):
        explicit = [{'start': 20, 'end': 30}]
        total_duration = 100
        clean = self.processor.get_clean_segments(explicit, total_duration)
        # Should be 0-20 and 30-100
        self.assertEqual(clean, [(0, 20), (30, 100)])

if __name__ == '__main__':
    unittest.main()
