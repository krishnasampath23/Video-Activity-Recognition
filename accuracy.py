import os
import glob
import keras
from keras_video import VideoFrameGenerator
import tensorflow as tf
import numpy as np

classes = [i.split(os.path.sep)[1] for i in glob.glob('UCF101/*')]
classes.sort()

SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 5
BS = 8

glob_pattern='UCF101/{classname}/*.avi'

data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

pattern = 'UCF101/{classname}/*.avi'
test = VideoFrameGenerator(
        glob_pattern=pattern,
        nb_frames=NBFRAME,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)

from keras.models import load_model

model = load_model('chkp_transfer_lstm_full/weights.192-1.04.hdf5')

loss,acc = model.evaluate(test)
print('************** Mobilenet UCF101 *********************')
print('\nLoss: {}, acc: {}\n'.format(loss, acc))

model2 = load_model('chkp_resnet_full/weights.110-0.100-0.96-0.38-0.94.hdf5') 
loss,acc = model2.evaluate(test)
print('************** Resnet UCF101 *********************')
print('\nLoss: {}, acc: {}\n'.format(loss, acc))