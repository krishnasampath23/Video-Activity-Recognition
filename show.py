import os
import glob
import keras
from keras_video import VideoFrameGenerator
import tensorflow as tf
import numpy as np
import sys


classes = [i.split(os.path.sep)[1] for i in glob.glob('test/*')]
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

pattern = str(sys.argv[1])

test2 = VideoFrameGenerator(
        glob_pattern=pattern,
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
print(classes[i])


import cv2
  
  
cap = cv2.VideoCapture(pattern)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))     
out = cv2.VideoWriter('Predictions/outpy.avi',
      cv2.VideoWriter_fourcc('M','J','P','G'),
       10,
       (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True: 
    
    # Write the frame into the file 'output.avi'
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, 
                str(classes[i]), 
                (0, 50), 
                font, 1, 
                (0, 0, 255), 
                4, 
                cv2.LINE_4)

    out.write(frame)

    # Display the resulting frame    
    cv2.imshow('frame',frame)

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break  

cap.release()
out.release()

cv2.destroyAllWindows()



# while(True):

#     ret, frame = cap.read()
  
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     cv2.putText(frame, 
#                 str(classes[i]), 
#                 (0, 50), 
#                 font, 1, 
#                 (0, 0, 255), 
#                 4, 
#                 cv2.LINE_4)
  
#     cv2.imshow('video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):

#         break



print('*************************************************')



print('*************************************************')