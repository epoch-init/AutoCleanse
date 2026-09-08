import cv2
import os
import subprocess
import json
import re
from PIL import Image
from utils import setup_temp_dir, cleanup_temp_dir

class VideoProcessor:
    def __init__(self, config, model_interface):
        self.config = config
        self.model_api = model_interface
        self.batch_size = 16 

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
            for _ in range(extract_distance - 1):
                vidcap.grab()
                processed_count += 1
                
            success, frame = vidcap.read()
            processed_count += 1
            
            if not success: break
                
            timestamp = processed_count / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_images.append(Image.fromarray(frame_rgb))
            batch_timestamps.append(timestamp)

            if len(batch_images) == self.batch_size:
                explicit_scenes, active_scene, buffer = self._process_batch(
                    batch_images, batch_timestamps, buffer, buffer_size, 
                    explicit_scenes, active_scene
                )
                batch_images = []
                batch_timestamps = []
                print(f"  Progress: {(processed_count/total_frames)*100:.1f}%", end="\r")

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

    def get_clean_segments(self, explicit_scenes, total_duration):
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
            print("No clean scenes found.")
            return clean_segments

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
        
        return clean_segments

    # --- SRT & Logging Utilities ---
    
    def parse_srt_time(self, time_str):
        h, m, s_ms = time_str.split(':')
        s, ms = s_ms.split(',')
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

    def format_srt_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms >= 1000:
            s += ms // 1000
            ms = ms % 1000
        if s >= 60:
            m += s // 60
            s = s % 60
        if m >= 60:
            h += m // 60
            m = m % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def map_time(self, t, clean_segments):
        """Maps an original timestamp to the new shortened timeline."""
        accumulated = 0.0
        for (seg_start, seg_end) in clean_segments:
            if t < seg_start:
                return accumulated
            if seg_start <= t <= seg_end:
                return accumulated + (t - seg_start)
            accumulated += (seg_end - seg_start)
        return accumulated

    def sync_srt(self, srt_path, output_srt_path, clean_segments):
        if not os.path.exists(srt_path):
            print(f"Error: SRT file not found at {srt_path}")
            return
            
        with open(srt_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        blocks = re.split(r'\n\s*\n', content.strip())
        new_blocks = []
        sub_idx = 1
        
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                time_line = lines[1]
                if '-->' not in time_line:
                    continue
                start_str, end_str = time_line.split('-->')
                start_t = self.parse_srt_time(start_str.strip())
                end_t = self.parse_srt_time(end_str.strip())
                text = '\n'.join(lines[2:])
                
                # Remap times
                new_start = self.map_time(start_t, clean_segments)
                new_end = self.map_time(end_t, clean_segments)
                
                # If the subtitle spans less than 100ms in the new timeline, it was cut
                if new_end - new_start > 0.1:
                    new_blocks.append(f"{sub_idx}\n{self.format_srt_time(new_start)} --> {self.format_srt_time(new_end)}\n{text}")
                    sub_idx += 1

        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(new_blocks))
        print(f"--- Clean Subtitles saved to: {output_srt_path} ---")

    def generate_summary(self, explicit_scenes, total_duration, output_path):
        removed_seconds = sum(max(0, scene['end'] - scene['start']) for scene in explicit_scenes)
        
        summary = {
            "original_duration_seconds": round(total_duration, 2),
            "new_duration_seconds": round(total_duration - removed_seconds, 2),
            "total_seconds_removed": round(removed_seconds, 2),
            "total_explicit_scenes_removed": len(explicit_scenes),
            "removed_segments": [
                {
                    "start": round(scene['start'], 2),
                    "end": round(scene['end'], 2),
                    "duration": round(scene['end'] - scene['start'], 2),
                    "label": "nudity"
                } for scene in explicit_scenes
            ]
        }
        
        base_name = os.path.splitext(output_path)[0]
        log_path = f"{base_name}_summary.json"
        
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"--- Processing Summary saved to: {log_path} ---")
