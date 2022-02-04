import os
import glob
import keras
from keras_video import VideoFrameGenerator
import tensorflow as tf
import numpy as np


classes = [i.split(os.path.sep)[1] for i in glob.glob('test/*')]
classes.sort()

SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 5
BS = 8


data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)


from keras.models import load_model

# model = load_model('chkp_transfer_lstm_full/weights.192-1.04.hdf5') #mobilent
model = load_model('chkp_resnet_full/weights.110-0.100-0.96-0.38-0.94.hdf5') #resnet

glob_pattern='Action_Type/1_Body_Motion/{classname}/*.avi'

pattern = 'Action_Type/1_Body_Motion/{classname}/*.avi'


test2 = VideoFrameGenerator(
        glob_pattern=pattern,
        nb_frames=NBFRAME,
        batch_size=1,
        # split=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)

model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=1)])
loss,acc = model.evaluate(test2)
print('\nAccuracy for Body Motion Only : {}\n'.format(acc))

print('**********************************************************')

glob_pattern='Action_Type/2_Human_human/{classname}/*.avi'

pattern = 'Action_Type/2_Human_human/{classname}/*.avi'

test2 = VideoFrameGenerator(
        glob_pattern=pattern,
        nb_frames=NBFRAME,
        batch_size=1,
        # split=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)

model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=1)])
loss,acc = model.evaluate(test2)
print('\nAccuracy for Human_Human_Interaction : {}\n'.format(acc))

print('**********************************************************')

glob_pattern = 'Action_Type/3_Human_object/{classname}/*.avi'

pattern = 'Action_Type/3_Human_object/{classname}/*.avi'


test2 = VideoFrameGenerator(
        glob_pattern=pattern,
        nb_frames=NBFRAME,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)


model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=1)])
loss,acc = model.evaluate(test2)
print('\nAccuracy for Human_Object_Interaction : {}\n'.format(acc))

print('**********************************************************')

glob_pattern = 'Action_Type/4_music/{classname}/*.avi'

pattern = 'Action_Type/4_music/{classname}/*.avi'


test2 = VideoFrameGenerator(
        glob_pattern=pattern,
        nb_frames=NBFRAME,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)


model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=1)])
loss,acc = model.evaluate(test2)
print('\nAccuracy for Playing_Muscial_Instruments : {}\n'.format(acc))

print('**********************************************************')

glob_pattern = 'Action_Type/5_sports/{classname}/*.avi'

pattern = 'Action_Type/5_sports/{classname}/*.avi'

test2 = VideoFrameGenerator(
        glob_pattern=pattern,
        nb_frames=NBFRAME,
        batch_size=1,
        # split=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)


model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=1)])
loss,acc = model.evaluate(test2)
print('\nAccuracy for Sports : {}\n'.format(acc))