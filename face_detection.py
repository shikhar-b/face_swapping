import cv2
import numpy as np
import face_recognition
from helpers import *
import pdb

def landmark_detect(source_image, target_image):
    face_landmarks_list_1 = face_recognition.face_landmarks(source_image)
    face_landmarks_list_2 = face_recognition.face_landmarks(target_image)

    # pdb.set_trace()

    if len(face_landmarks_list_1) == 0 or len(face_landmarks_list_2) == 0:
        return []

    points_1, points_2 = intersect(face_landmarks_list_1[0], face_landmarks_list_2[0])
    #visualizeFeatures(img, (values[:, 0], values[:, 1]))
    points_1 = np.array(points_1).astype(np.int32)
    points_2 = np.array(points_2).astype(np.int32)

    return listOfListToTuples(points_1.tolist()), listOfListToTuples(points_2.tolist())

def landmark_detect_clahe(source_image, target_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    source_gray = convert_BGR2Gray(source_image)
    target_gray = convert_BGR2Gray(target_image)

    cl1 = clahe.apply(source_gray)
    cl2 = clahe.apply(target_gray)

    landmark_detect(cl1, cl2)

def intersect(face_landmarks_1, face_landmarks_2):
    points_1,points_2 = [],[]

    for key_1, value_1 in face_landmarks_1.items():
        if key_1 in face_landmarks_2:
            value_2 = face_landmarks_2[key_1]
            if len(value_1) == len(value_2):
                points_1.extend(value_1)
                points_2.extend(value_2)

    return points_1,points_2
