import argparse
import os

from utils import get_config
from processor import VideoProcessor
from models.adapter_api import ModelInterface
from models.nsfw_model import NsfwModelAdapter

def main():
    parser = argparse.ArgumentParser(description="AutoCleanse: Automatic NSFW Scene Remover")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--srt", required=False, help="Optional: Path to subtitle file to sync")
    
    args = parser.parse_args()
    config = get_config()

    print("--- Initializing AutoCleanse ---")
    # 1. Load Model
    nsfw_adapter = NsfwModelAdapter(threshold=config['threshold'])
    nsfw_adapter.load_model()
    
    # 2. Setup Interface
    model_api = ModelInterface([nsfw_adapter])
    
    # 3. Process Video
    processor = VideoProcessor(config, model_api)
    explicit_scenes, duration = processor.analyze_video(args.input)
    
    # 4. Final Cut
    clean_segments = processor.cut_video(args.input, args.output, explicit_scenes, duration)
    
    # 5. Output Summary
    processor.generate_summary(explicit_scenes, duration, args.output)

    # 6. Process Subtitles if provided
    if args.srt:
        base_name = os.path.splitext(args.output)[0]
        output_srt = f"{base_name}.srt"
        processor.sync_srt(args.srt, output_srt, clean_segments)
        
    print("--- Processing Fully Completed ---")

if __name__ == '__main__':
    main()
