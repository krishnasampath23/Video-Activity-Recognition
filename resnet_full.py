import os
import glob
import keras
import tensorflow as tf
from keras_video import VideoFrameGenerator

classes = [i.split(os.path.sep)[1] for i in glob.glob('train/*')]
classes.sort()

SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 5
BS = 8

glob_pattern='train/{classname}/*.avi'

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



def build_resnet(shape=(224, 224, 3)):
    model = tf.keras.applications.ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classifier_activation="softmax",)

    trainable = 24

    for layer in model.layers[:-trainable]:
        layer.trainable = False
    for layer in model.layers[-trainable:]:
        layer.trainable = True
    output = keras.layers.GlobalMaxPool2D()
    return keras.Sequential([model, output])


def action_model(shape=(5, 224, 224, 3), nbout=101):

    convnet = build_resnet(shape[1:])
    
    model = keras.Sequential()
    model.add(TimeDistributed(convnet, input_shape=shape))

    model.add(GRU(128))

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


model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS=150

callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp_resnet_full_gru/weights.{epoch:02d}-{loss:.3f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
        verbose=1),
]
model.fit(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)

print('***********************************************************')

model.save('model_resnet_full')

print('***********************************************************') 