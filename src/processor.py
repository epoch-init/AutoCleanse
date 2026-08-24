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
        extract_distance = max(1, int(fps / self.config['fps']))
        
        explicit_scenes = []
        active_scene = None
        
        # SMOOTHING BUFFER: Requires 3 consecutive hits to trigger
        buffer = []
        buffer_size = 3 

        print(f"--- Scanning: {os.path.basename(video_path)} (Smoothing Enabled) ---")
        
        count = 0
        while True:
            success, frame = vidcap.read()
            if not success: break
            
            if count % extract_distance == 0:
                timestamp = count / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                results = self.model_api.analyze_frame(pil_img)
                score = max(results.values())
                
                # Check if current frame is suspicious
                is_hit = score >= self.config['threshold']
                buffer.append(is_hit)
                if len(buffer) > buffer_size: buffer.pop(0)

                # TRIGGER LOGIC: True only if the majority of the buffer is True
                is_explicit = sum(buffer) >= 2 

                if is_explicit:
                    if active_scene is None:
                        active_scene = {'start': timestamp, 'end': timestamp}
                        print(f"  [!] NSFW Scene Started: {timestamp:.2f}s")
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
        if active_scene: explicit_scenes.append(active_scene)

        return self.merge_and_pad_scenes(explicit_scenes), duration

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

        print(f"\nCreating clean version...")
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

        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        cleanup_temp_dir(self.config['temp_dir'])
        print(f"--- Clean video: {output_path} ---")
