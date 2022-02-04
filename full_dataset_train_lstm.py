import os
import glob
import keras
import tensorflow as tf
from keras_video import VideoFrameGenerator

classes = [i.split(os.path.sep)[1] for i in glob.glob('UCF101/*')]
classes.sort()
print(classes)

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


from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D

from keras.layers import TimeDistributed, GRU, Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam



def build_mobilenet(shape=(224, 224, 3), nbout=3):
    model = keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=shape,
        weights='imagenet')
    # Keep 9 layers to train﻿﻿
    trainable = 9
    for layer in model.layers[:-trainable]:
        layer.trainable = False
    for layer in model.layers[-trainable:]:
        layer.trainable = True
    output = keras.layers.GlobalMaxPool2D()
    return keras.Sequential([model, output])


def action_model(shape=(5, 112, 112, 3), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_mobilenet(shape[1:])
    
    # then create our final model
    model = keras.Sequential()
    model.add(TimeDistributed(convnet, input_shape=shape))

    # model.add(LSTM(128,input_shape=shape[1:]))

    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(.2))

    # model.add(LSTM(128))

    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(.2))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(.2))

    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(nbout, activation='softmax'))

    model.add(LSTM(128))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))

    return model

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 224, 224, 3)
model = action_model(INSHAPE, len(classes))
optimizer = keras.optimizers.SGD()
# model.compile(
#     optimizer=Adam(lr=0.001, decay=1e-6),
#     loss='categorical_crossentropy',
#     metrics=['acc']
# )

model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS=2500

callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp_transfer_lstm_full/weights.{epoch:02d}-{loss}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
        verbose=1)
]
model.fit(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)

print('***********************************************************')

model.save('model_transfer_lstm_full')

print('***********************************************************') 