import cv2
import os
import subprocess
from PIL import Image
from utils import setup_temp_dir, cleanup_temp_dir

class VideoProcessor:
    def __init__(self, config, model_interface):
        self.config = config
        self.model_api = model_interface

    def analyze_video(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Extract frames based on config FPS (e.g. 2.0 means every 0.5s)
        extract_distance = max(1, int(fps / self.config['fps']))
        
        explicit_scenes = []
        active_scene = None
        count = 0

        print(f"--- AutoCleanse: Scanning {os.path.basename(video_path)} ---")
        
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            
            if count % extract_distance == 0:
                timestamp = count / fps
                
                # Faster: Convert OpenCV BGR to RGB and then to PIL in RAM
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Send PIL image directly to CLIP
                results = self.model_api.analyze_frame(pil_img)
                score = max(results.values())
                
                is_explicit = score >= self.config['threshold']

                if is_explicit:
                    if active_scene is None:
                        active_scene = {'start': timestamp, 'end': timestamp}
                        print(f"  [!] Flagged: {timestamp:.2f}s (Score: {score:.2f})")
                    else:
                        active_scene['end'] = timestamp
                else:
                    if active_scene is not None:
                        explicit_scenes.append(active_scene)
                        active_scene = None

            count += 1
            if count % (extract_distance * 10) == 0:
                print(f"  Progress: {(count/total_frames)*100:.1f}%", end="\r")

        vidcap.release()
        if active_scene:
            explicit_scenes.append(active_scene)

        processed_scenes = self.merge_and_pad_scenes(explicit_scenes)
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

    def cut_video(self, input_path, output_path, explicit_scenes, total_duration):
        # We find the clean segments (inverted NSFW scenes)
        clean_segments = []
        last_end = 0
        for scene in explicit_scenes:
            if scene['start'] > last_end:
                clean_segments.append((last_end, scene['start']))
            last_end = scene['end']
        if last_end < total_duration:
            clean_segments.append((last_end, total_duration))

        if not clean_segments:
            print("No clean scenes found.")
            return

        setup_temp_dir(self.config['temp_dir'])
        concat_list = os.path.join(self.config['temp_dir'], "concat_list.txt")
        segment_files = []

        print(f"\nStitching {len(clean_segments)} clean segments together...")
        for i, (start, end) in enumerate(clean_segments):
            segment_path = os.path.join(self.config['temp_dir'], f"seg_{i}.mp4")
            duration = end - start
            codec = "libx264" if self.config['re_encode'] else "copy"
            
            cmd = [
                'ffmpeg', '-y', '-ss', str(start), '-i', input_path,
                '-t', str(duration), '-c', codec, '-avoid_negative_ts', '1', segment_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            segment_files.append(f"file '{os.path.abspath(segment_path)}'\n")

        with open(concat_list, "w") as f:
            f.writelines(segment_files)

        final_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list,
            '-c', 'copy', output_path
        ]
        subprocess.run(final_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        cleanup_temp_dir(self.config['temp_dir'])
        print(f"--- Clean video created: {output_path} ---")
