import cv2
import os
from utils import setup_temp_dir, cleanup_temp_dir

class VideoProcessor:
    def __init__(self, config, model_interface):
        self.config = config
        self.model_api = model_interface

    def extract_frames(self, video_path):
        setup_temp_dir(self.config['temp_dir'])
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        # We extract frames based on the config FPS (e.g., 0.2 means 1 frame every 5 seconds)
        extract_distance = int(fps / self.config['fps'])
        
        count = 0
        extracted_paths = []
        
        while True:
            success, image = vidcap.read()
            if not success:
                break
            
            if count % extract_distance == 0:
                frame_time = count / fps
                frame_path = os.path.join(self.config['temp_dir'], f"frame_{frame_time:.2f}.jpg")
                cv2.imwrite(frame_path, image)
                extracted_paths.append((frame_time, frame_path))
            
            count += 1
            
        vidcap.release()
        return extracted_paths

    def analyze_video(self, video_path):
        frames = self.extract_frames(video_path)
        explicit_scenes = []
        
        active_scene = None

        for timestamp, path in frames:
            results = self.model_api.analyze_frame(path)
            # Check if any model (currently just nudity) exceeds threshold
            is_explicit = any(score >= self.config['threshold'] for score in results.values())

            if is_explicit:
                if active_scene is None:
                    active_scene = {'start': timestamp, 'end': timestamp, 'label': 'nudity'}
                else:
                    active_scene['end'] = timestamp
            else:
                if active_scene is not None:
                    # Apply min_delta_between_splits logic
                    explicit_scenes.append(active_scene)
                    active_scene = None

        # Handle final scene
        if active_scene:
            explicit_scenes.append(active_scene)

        return self.merge_and_pad_scenes(explicit_scenes)

    def merge_and_pad_scenes(self, scenes):
        if not scenes: return []
        
        merged = []
        if not scenes: return merged

        curr = scenes[0]
        for next_scene in scenes[1:]:
            if next_scene['start'] - curr['end'] < self.config['min_delta_between_splits']:
                curr['end'] = next_scene['end']
            else:
                merged.append(self.apply_padding(curr))
                curr = next_scene
        merged.append(self.apply_padding(curr))
        
        return merged

    def apply_padding(self, scene):
        scene['start'] = max(0, scene['start'] - self.config['padding'])
        scene['end'] = scene['end'] + self.config['padding']
        return scene
