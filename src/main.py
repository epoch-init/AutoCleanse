import argparse
from utils import get_config
from processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="AutoCleanse: Automatic NSFW Scene Remover")
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--output", required=True, help="Path to save the cleaned video")
    
    args = parser.parse_args()
    config = get_config()

    print(f"--- AutoCleanse Starting ---")
    print(f"Input: {args.input}")
    print(f"Config: {config}")

    # Logic will be implemented in Phase 2/3
    # processor = VideoProcessor(config)
    # processor.process(args.input, args.output)

if __name__ == '__main__':
    main()
