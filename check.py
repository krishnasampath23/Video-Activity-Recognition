import os
import glob
import keras
from keras_video import VideoFrameGenerator
import tensorflow as tf
import numpy as np

classes = [i.split(os.path.sep)[1] for i in glob.glob('vid/*')]
classes.sort()

SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 5
BS = 8

glob_pattern='vid/{classname}/*.avi'

data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

train = VideoFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split=.33,
    shuffle=True,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=True)

valid = train.get_validation_generator()


from keras.models import load_model
# model = load_model('chkp100/weights.200-0.61.hdf5')
model = load_model('chkp_resnet_full/weights.110-0.100-0.96-0.38-0.94.hdf5') 

# model = load_model('model_transfer/')

# model = tf.keras.models.load_model('weights.100-0.48.hdf5')

print('*************************************************')
print('*************************************************')
print(classes)
print('*************************************************')
print('*************************************************')


pattern2 = 'v_TrampolineJumping_g07_c03.avi'
test2 = VideoFrameGenerator(
        glob_pattern=pattern2,
        nb_frames=NBFRAME,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)

preds2 = model.predict(test2)
print('*************************************************')
print(preds2)
print(np.argmax(preds2))
i = np.argmax(preds2)
if i == 0:
    print("Prediction: Archery " + str(round(preds2[0][i]*100,2)) + "%")
if i == 1:
    print("Prediction: CricketShot " + str(round(preds2[0][i]*100,2)) + "%")
if i == 2:
    print("Prediction: Trampoline Jumping " + str(round(preds2[0][i]*100,2)) + "%")
if i == 3:
    print("Prediction: Walking With Dog " + str(round(preds2[0][i]*100,2)) + "%")


print('*************************************************')

pattern2 = 'v_CricketShot_g14_c01.avi'
test2 = VideoFrameGenerator(
        glob_pattern=pattern2,
        nb_frames=NBFRAME,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)

preds2 = model.predict(test2)
print('*************************************************')
print(preds2)
print(np.argmax(preds2))
i = np.argmax(preds2)
if i == 0:
    print("Prediction: Archery " + str(round(preds2[0][i]*100,2)) + "%")
if i == 1:
    print("Prediction: CricketShot " + str(round(preds2[0][i]*100,2)) + "%")
if i == 2:
    print("Prediction: Trampoline Jumping " + str(round(preds2[0][i]*100,2)) + "%")
if i == 3:
    print("Prediction: Walking With Dog " + str(round(preds2[0][i]*100,2)) + "%")

print('*************************************************')

pattern2 = 'v_Archery_g24_c03.avi'
test2 = VideoFrameGenerator(
        glob_pattern=pattern2,
        nb_frames=NBFRAME,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)

preds2 = model.predict(test2)
print(preds2)
print(np.argmax(preds2))
i = np.argmax(preds2)
if i == 0:
    print("Prediction: Archery " + str(round(preds2[0][i]*100,2)) + "%")
if i == 1:
    print("Prediction: CricketShot " + str(round(preds2[0][i]*100,2)) + "%")
if i == 2:
    print("Prediction: Trampoline Jumping " + str(round(preds2[0][i]*100,2)) + "%")
if i == 3:
    print("Prediction: Walking With Dog " + str(round(preds2[0][i]*100,2)) + "%")


print('*************************************************')

pattern2 = 'v_WalkingWithDog_g05_c05.avi'
test2 = VideoFrameGenerator(
        glob_pattern=pattern2,
        nb_frames=NBFRAME,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)

preds2 = model.predict(test2)
print(preds2)
print(np.argmax(preds2))
i = np.argmax(preds2)
if i == 0:
    print("Prediction: Archery " + str(round(preds2[0][i]*100,2)) + "%")
if i == 1:
    print("Prediction: CricketShot " + str(round(preds2[0][i]*100,2)) + "%")
if i == 2:
    print("Prediction: Trampoline Jumping " + str(round(preds2[0][i]*100,2)) + "%")
if i == 3:
    print("Prediction: Walking With Dog " + str(round(preds2[0][i]*100,2)) + "%")

print('*************************************************')



print('*************************************************')