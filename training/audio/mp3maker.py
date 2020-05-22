
import subprocess

import os
#The trailers files
VIDEOS_PATH = 
#video format
VIDEOS_EXTENSION = '.mp4' 

os.chdir(VIDEOS_PATH)

files = os.listdir(VIDEOS_PATH)

for f in files:
    if not f.endswith(VIDEOS_EXTENSION):
        continue
    command =  'ffmpeg -i '+VIDEOS_PATH+'/'+f+' '+'results/'+f[:-4]+'.mp3'
    # You can change the location the film's audio is going to be stored 
    subprocess.call(command, shell=True)