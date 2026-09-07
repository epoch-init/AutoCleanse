import cv2
import os
import subprocess
from PIL import Image
from utils import setup_temp_dir, cleanup_temp_dir

class VideoProcessor:
    def __init__(self, config, model_interface):
        self.config = config
        self.model_api = model_interface
        self.batch_size = 16 # Process 16 frames simultaneously (Fits easily in 2GB VRAM)

    def analyze_video(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        extract_distance = max(1, int(fps / self.config['fps']))
        
        explicit_scenes = []
        active_scene = None
        
        buffer = []
        buffer_size = 3 

        print(f"--- Fast Scanning: {os.path.basename(video_path)} ---")
        
        batch_images = []
        batch_timestamps = []
        processed_count = 0

        while True:
            # FAST SKIP: grab() skips the expensive decoding phase for frames we ignore
            for _ in range(extract_distance - 1):
                vidcap.grab()
                processed_count += 1
                
            # Actually read and decode the target frame
            success, frame = vidcap.read()
            processed_count += 1
            
            if not success: 
                break
                
            timestamp = processed_count / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_images.append(Image.fromarray(frame_rgb))
            batch_timestamps.append(timestamp)

            # If batch is full, send to GPU
            if len(batch_images) == self.batch_size:
                explicit_scenes, active_scene, buffer = self._process_batch(
                    batch_images, batch_timestamps, buffer, buffer_size, 
                    explicit_scenes, active_scene
                )
                # Clear batch
                batch_images = []
                batch_timestamps = []
                
                print(f"  Progress: {(processed_count/total_frames)*100:.1f}%", end="\r")

        # Process any remaining frames in the final partial batch
        if batch_images:
            explicit_scenes, active_scene, buffer = self._process_batch(
                batch_images, batch_timestamps, buffer, buffer_size, 
                explicit_scenes, active_scene
            )

        vidcap.release()
        if active_scene: explicit_scenes.append(active_scene)

        print("\n--- Scan Complete ---")
        return self.merge_and_pad_scenes(explicit_scenes), duration

    def _process_batch(self, images, timestamps, buffer, buffer_size, explicit_scenes, active_scene):
        """Helper method to process a batch and update state variables."""
        batch_results = self.model_api.analyze_batch(images)
        
        for i, results in enumerate(batch_results):
            score = max(results.values())
            timestamp = timestamps[i]
            
            is_hit = score >= self.config['threshold']
            buffer.append(is_hit)
            if len(buffer) > buffer_size: buffer.pop(0)

            is_explicit = sum(buffer) >= 2 

            if is_explicit:
                if active_scene is None:
                    active_scene = {'start': timestamp, 'end': timestamp}
                    print(f"  [!] NSFW Scene Started: {timestamp:.2f}s (Score: {score:.2f})")
                else:
                    active_scene['end'] = timestamp
            else:
                if active_scene is not None:
                    explicit_scenes.append(active_scene)
                    active_scene = None
                    
        return explicit_scenes, active_scene, buffer

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
