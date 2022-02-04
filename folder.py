import os
import glob
import keras
from keras_video import VideoFrameGenerator
import tensorflow as tf
import numpy as np

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        

# Example
classes = [i.split(os.path.sep)[1] for i in glob.glob('test/*')]
classes.sort()
for i in classes:
	createFolder('Empty/{i}/')
# Creates a folder in the current directory called data