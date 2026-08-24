import json


def get_config():
  config = json.load(open("config.json"))
  print(config)
  
  
if __name__ == '__main__':
  get_config() 
