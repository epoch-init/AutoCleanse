import argparse
import sys
import os

from utils import get_config
from processor import VideoProcessor
from models.adapter_api import ModelInterface
from models.nsfw_model import NsfwModelAdapter

def main():
    parser = argparse.ArgumentParser(description="AutoCleanse: Automatic NSFW Scene Remover")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    
    args = parser.parse_args()
    config = get_config()

    # 1. Load Model
    nsfw_adapter = NsfwModelAdapter(threshold=config['threshold'])
    nsfw_adapter.load_model()
    
    # 2. Setup Interface
    model_api = ModelInterface([nsfw_adapter])
    
    # 3. Process Video
    processor = VideoProcessor(config, model_api)
    explicit_scenes, duration = processor.analyze_video(args.input)
    
    # 4. Final Cut
    processor.cut_video(args.input, args.output, explicit_scenes, duration)

if __name__ == '__main__':
    main()
