import os
os.system('pip3 install -r requirements.txt')

os.system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/')

os.system('python save_model.py --model yolov4')

os.system('python object_tracker.py --video ./data/video/rohan_walking.mp4 --output ./outputs/tracker.avi --model yolov4 --info')

# define helper function to display videos
import io 
from IPython.display import HTML
from base64 import b64encode
def show_video(file_name, width=640):
  # show resulting deepsort video
  mp4 = open(file_name,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML("""
  <video width="{0}" controls>
        <source src="{1}" type="video/mp4">
  </video>
  """.format(width, data_url))


  # convert resulting video from avi to mp4 file format

path_video = os.path.join("outputs","tracker.avi")

os.system('ffmpeg -y -loglevel panic -i outputs/tracker.avi outputs/output.mp4')


# output object tracking video
path_output = os.path.join("outputs","output.mp4")
#show_video(path_output, width=960)