import cv2
import numpy as np
import face_recognition
from helpers import *
import pdb

def landmark_detect(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    #pdb.set_trace()
    values = []
    for key, value in face_landmarks_list[0].items():
        for ent in value:
            values.append(ent)

    values = np.array(values).astype(np.int32)
    #visualizeFeatures(img, (values[:, 0], values[:, 1]))
    #values is 72*2

    values = listOfListToTuples(values)
    return values
