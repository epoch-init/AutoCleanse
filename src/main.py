import argparse
from utils import get_config
from processor import VideoProcessor
from models.adapter_api import ModelInterface
from models.nsfw_model import NsfwModelAdapter

def main():
    parser = argparse.ArgumentParser(description="AutoCleanse: Automatic NSFW Scene Remover")
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--output", required=True, help="Path to save the cleaned video")
    
    args = parser.parse_args()
    config = get_config()

    # Initialize Model
    nsfw_adapter = NsfwModelAdapter(threshold=config['threshold'])
    nsfw_adapter.load_model()
    
    # Initialize API and Processor
    model_api = ModelInterface([nsfw_adapter])
    processor = VideoProcessor(config, model_api)

    print(f"--- AutoCleanse Started ---")
    
    # Step 1: Analyze
    explicit_scenes, duration = processor.analyze_video(args.input)
    
    # Step 2: Cut and Stitch
    processor.cut_video(args.input, args.output, explicit_scenes, duration)

if __name__ == '__main__':
    main()
