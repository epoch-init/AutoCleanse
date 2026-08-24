import cv2
import os
import subprocess
import json
from utils import setup_temp_dir, cleanup_temp_dir

class VideoProcessor:
    def __init__(self, config, model_interface):
        self.config = config
        self.model_api = model_interface

    def extract_frames(self, video_path):
        setup_temp_dir(self.config['temp_dir'])
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        extract_distance = int(fps / self.config['fps'])
        
        count = 0
        extracted_paths = []
        
        print(f"Extracting frames at {self.config['fps']} FPS...")
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
        return extracted_paths, duration

    def analyze_video(self, video_path):
        frames, duration = self.extract_frames(video_path)
        explicit_scenes = []
        active_scene = None

        print(f"Analyzing {len(frames)} frames for explicit content...")
        for timestamp, path in frames:
            results = self.model_api.analyze_frame(path)
            is_explicit = any(score >= self.config['threshold'] for score in results.values())

            if is_explicit:
                if active_scene is None:
                    active_scene = {'start': timestamp, 'end': timestamp, 'label': 'nudity'}
                else:
                    active_scene['end'] = timestamp
            else:
                if active_scene is not None:
                    explicit_scenes.append(active_scene)
                    active_scene = None

        if active_scene:
            explicit_scenes.append(active_scene)

        processed_scenes = self.merge_and_pad_scenes(explicit_scenes)
        
        # Save metadata for transparency
        with open("detection_log.json", "w+") as f:
            json.dump(processed_scenes, f, indent=2)
            
        return processed_scenes, duration

    def merge_and_pad_scenes(self, scenes):
        if not scenes: return []
        
        merged = []
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

    def get_clean_segments(self, explicit_scenes, total_duration):
        """Inverts explicit scenes to find the clean parts of the video."""
        clean_segments = []
        last_end = 0

        for scene in explicit_scenes:
            if scene['start'] > last_end:
                clean_segments.append((last_end, scene['start']))
            last_end = scene['end']

        if last_end < total_duration:
            clean_segments.append((last_end, total_duration))
        
        return clean_segments

    def cut_video(self, input_path, output_path, explicit_scenes, total_duration):
        clean_segments = self.get_clean_segments(explicit_scenes, total_duration)
        
        if not clean_segments:
            print("No clean scenes found. Entire video is flagged.")
            return

        # Create a list file for ffmpeg concat
        concat_list = os.path.join(self.config['temp_dir'], "concat_list.txt")
        segment_files = []

        print(f"Cutting {len(clean_segments)} clean segments...")
        for i, (start, end) in enumerate(clean_segments):
            segment_path = os.path.join(self.config['temp_dir'], f"seg_{i}.mp4")
            duration = end - start
            
            # FFmpeg command to cut
            codec = "libx264" if self.config['re_encode'] else "copy"
            cmd = [
                'ffmpeg', '-y', '-ss', str(start), '-i', input_path,
                '-t', str(duration), '-c', codec, '-avoid_negative_ts', '1', segment_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            segment_files.append(f"file '{os.path.abspath(segment_path)}'\n")

        with open(concat_list, "w") as f:
            f.writelines(segment_files)

        # Concatenate segments
        print("Stitching segments together...")
        concat_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list,
            '-c', 'copy', output_path
        ]
        subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        cleanup_temp_dir(self.config['temp_dir'])
        print(f"Done! Clean video saved to: {output_path}")
