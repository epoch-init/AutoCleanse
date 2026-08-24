import json
import os
import shutil

def get_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def setup_temp_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

def cleanup_temp_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
