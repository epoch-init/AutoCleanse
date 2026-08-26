import cv2
import os
import subprocess
from PIL import Image
from utils import setup_temp_dir, cleanup_temp_dir

class VideoProcessor:
    def __init__(self, config, model_interface):
        self.config = config
        self.model_api = model_interface
        self.batch_size = 12 # Optimized for 2GB VRAM

    def analyze_video(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        extract_distance = max(1, int(fps / self.config['fps']))
        
        explicit_scenes = []
        active_scene = None
        
        # Smoothing
        score_history = []
        
        print(f"--- Fast Scan: {os.path.basename(video_path)} ---")
        
        count = 0
        batch_frames = []
        batch_timestamps = []

        while True:
            success, frame = vidcap.read()
            if not success: break
            
            if count % extract_distance == 0:
                timestamp = count / fps
                
                # Fast Resize before PIL conversion
                small_frame = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                batch_frames.append(Image.fromarray(frame_rgb))
                batch_timestamps.append(timestamp)

                # Process batch when full
                if len(batch_frames) >= self.batch_size:
                    self._process_batch(batch_frames, batch_timestamps, score_history, explicit_scenes, active_scene)
                    # Note: active_scene is passed by reference, but we need to track it
                    if explicit_scenes and active_scene == explicit_scenes[-1]:
                        active_scene = explicit_scenes[-1]
                    
                    # Update local state tracker
                    if active_scene and (not explicit_scenes or active_scene != explicit_scenes[-1]):
                        pass # scene still active
                    elif explicit_scenes and not active_scene:
                        pass # scene just ended

                    batch_frames = []
                    batch_timestamps = []

            count += 1
            if count % (extract_distance * 10) == 0:
                print(f"  Progress: {(count/total_frames)*100:.1f}%", end="\r")

        # Process remaining
        if batch_frames:
            self._process_batch(batch_frames, batch_timestamps, score_history, explicit_scenes, active_scene)

        vidcap.release()
        
        # Cleanup logic for final scene
        final_scenes = self.merge_and_pad_scenes(explicit_scenes)
        return final_scenes, duration

    def _process_batch(self, frames, timestamps, history, explicit_list, active_ref):
        # We access the model's batch predictor via the unified API
        # For simplicity, we directly call nsfw_model here or through interface
        adapter = self.model_api.adapters[0] 
        scores = adapter.predict_batch(frames)

        for i, score in enumerate(scores):
            t = timestamps[i]
            history.append(score)
            if len(history) > 5: history.pop(0)
            
            avg_score = sum(history) / len(history)
            
            # Weighted Decision
            is_explicit = avg_score >= self.config['threshold']

            # Update the explicit_list directly
            if is_explicit:
                if not explicit_list or 'ended' in explicit_list[-1]:
                    explicit_list.append({'start': t, 'end': t})
                else:
                    explicit_list[-1]['end'] = t
            else:
                if explicit_list and 'ended' not in explicit_list[-1]:
                    explicit_list[-1]['ended'] = True # Mark as finished

    def merge_and_pad_scenes(self, scenes):
        if not scenes: return []
        # Remove the 'ended' helper flag
        for s in scenes: s.pop('ended', None)
        
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

        if not clean_segments: return
        setup_temp_dir(self.config['temp_dir'])
        concat_list = os.path.join(self.config['temp_dir'], "concat_list.txt")
        segment_files = []

        print(f"\nFinal Stitching...")
        for i, (start, end) in enumerate(clean_segments):
            segment_path = os.path.join(self.config['temp_dir'], f"seg_{i}.mp4")
            cmd = ['ffmpeg', '-y', '-ss', str(start), '-i', input_path, '-t', str(end-start), '-c', 'copy', '-avoid_negative_ts', '1', segment_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            segment_files.append(f"file '{os.path.abspath(segment_path)}'\n")

        with open(concat_list, "w") as f: f.writelines(segment_files)
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        cleanup_temp_dir(self.config['temp_dir'])
        print(f"--- Processed Video Saved: {output_path} ---")
