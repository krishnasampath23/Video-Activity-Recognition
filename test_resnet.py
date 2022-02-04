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

glob_pattern='test/{classname}/*.avi'

data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)


from keras.models import load_model

# model = load_model('chkp_transfer_lstm_full/weights.192-1.04.hdf5') #mobilent
model = load_model('chkp_resnet_full/weights.110-0.100-0.96-0.38-0.94.hdf5') #resnet

# model = load_model('model_transfer/')

# model = tf.keras.models.load_model('weights.100-0.48.hdf5')

print('*************************************************')
print('*************************************************')
# print(classes)
print('*************************************************')
print('*************************************************')


pattern = 'test/{classname}/*.avi'
test2 = VideoFrameGenerator(
        glob_pattern=pattern,
        nb_frames=NBFRAME,
        batch_size=1,
        # split=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)
# valid = train.get_validation_generator()


model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=10)])
loss,acc = model.evaluate(test2)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=7)])
loss,acc = model.evaluate(test2)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
loss,acc = model.evaluate(test2)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=3)])
loss,acc = model.evaluate(test2)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


model.compile(metrics= [tf.keras.metrics.TopKCategoricalAccuracy(k=1)])
loss,acc = model.evaluate(test2)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

# s = 0
# u = 0
# m = 0
# pred = model.predict(test2)
# for i in pred:
#     j = np.argmax(i)
#     top3 = i.argsort()[-3:][::-1]
#     top5 = i.argsort()[-5:][::-1]

#     # print(classes[j])
#     # print(i[j])

#     classname=
#     # for y in pattern[4:]:
#     #     if y == '/':
#     #         break
#     #     classname=classname+y
#     print(pattern)
#     print(classname)
#     break

#     index = 0
#     for z in classes:
#         if z == classname:
#             break
#         index = index+1 

#     if j == index:
#         s=s+1
#     if index in top3:
#         u=u+1
#     if index in top5:
#         m=m+1

print('*************************************************')

# acc = (s/3775)*100
# print(acc)
# acc = (u/3775)*100
# print(acc)
# acc = (m/3775)*100
# print(acc)

print('*************************************************')
print('*************************************************')
# pattern2 = 'v_TrampolineJumping_g07_c03.avi'
# test2 = VideoFrameGenerator(
#         glob_pattern=pattern2,
#         nb_frames=NBFRAME,
#         batch_size=1,
#         target_shape=SIZE,
#         nb_channel=CHANNELS,
#         transformation=data_aug,
#         use_frame_cache=True)

# preds2 = model.predict(test2)
# print('*************************************************')
# # print(preds2)

# i = np.argmax(preds2)
# print(classes[i])
# print(preds2[0][i])


# print('*************************************************')

# pattern2 = 'v_CricketShot_g14_c01.avi'
# test2 = VideoFrameGenerator(
#         glob_pattern=pattern2,
#         nb_frames=NBFRAME,
#         batch_size=1,
#         target_shape=SIZE,
#         nb_channel=CHANNELS,
#         transformation=data_aug,
#         use_frame_cache=True)

# preds2 = model.predict(test2)
# print('*************************************************')
# i = np.argmax(preds2)
# print(classes[i])
# print(preds2[0][i])

# print('*************************************************')

# pattern2 = 'v_Archery_g24_c03.avi'
# test2 = VideoFrameGenerator(
#         glob_pattern=pattern2,
#         nb_frames=NBFRAME,
#         batch_size=1,
#         target_shape=SIZE,
#         nb_channel=CHANNELS,
#         transformation=data_aug,
#         use_frame_cache=True)

# preds2 = model.predict(test2)
# i = np.argmax(preds2)
# print(classes[i])
# print(preds2[0][i])

# print('*************************************************')

# pattern2 = 'v_WalkingWithDog_g05_c05.avi'
# test2 = VideoFrameGenerator(
#         glob_pattern=pattern2,
#         nb_frames=NBFRAME,
#         batch_size=1,
#         target_shape=SIZE,
#         nb_channel=CHANNELS,
#         transformation=data_aug,
#         use_frame_cache=True)

# preds2 = model.predict(test2)
# i = np.argmax(preds2)
# print(classes[i])
# print(preds2[0][i])

# print('*************************************************')



# print('*************************************************')