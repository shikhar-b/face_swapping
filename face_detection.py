import cv2
import numpy as np
import face_recognition
from helpers import *
import pdb, logging
import dlib

PREDICTOR_PATH = "datasets/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        print 'TooManyFaces'
    if len(rects) == 0:
        print 'NoFaces'

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def landmark_detect_dlib(source_image, target_image, frame_number):
     test = get_landmarks(target_image)
     visualizeFeatures(target_image, test)

def landmark_detect(source_image, target_image, frame_number):
    face_landmarks_list_1 = face_recognition.face_landmarks(source_image)
    face_landmarks_list_2 = face_recognition.face_landmarks(target_image)

    # pdb.set_trace()

    if len(face_landmarks_list_1) == 0 or len(face_landmarks_list_2) == 0:
        if len(face_landmarks_list_1) == 0 and len(face_landmarks_list_2) == 0:
            logging.error('Frame: ' + str(frame_number) + ' - face not detected in both')
        if len(face_landmarks_list_1) == 0:
            logging.error('Frame: ' + str(frame_number) + ' - face not detected in source')
        if len(face_landmarks_list_2) == 0:
            logging.error('Frame: ' + str(frame_number) + ' - face not detected in target')
        return []

    points_1, points_2 = intersect(face_landmarks_list_1[0], face_landmarks_list_2[0])
    #visualizeFeatures(img, (values[:, 0], values[:, 1]))
    points_1 = np.array(points_1).astype(np.int32)
    points_2 = np.array(points_2).astype(np.int32)


    return face_landmarks_list_1[0], face_landmarks_list_2[0], listOfListToTuples(points_1.tolist()), listOfListToTuples(points_2.tolist())

def landmark_detect_dual(source_image, target_image, frame_number):
    face_landmarks_list_1 = face_recognition.face_landmarks(source_image)
    face_landmarks_list_2 = face_recognition.face_landmarks(target_image)
    face_landmarks_list_2.append(face_landmarks_list_2.pop(0))

    if len(face_landmarks_list_1) == 0 or len(face_landmarks_list_2) == 0:
        if len(face_landmarks_list_1) == 0 and len(face_landmarks_list_2) == 0:
            logging.error('Frame: ' + str(frame_number) + ' - face not detected in both')
        if len(face_landmarks_list_1) == 0:
            logging.error('Frame: ' + str(frame_number) + ' - face not detected in source')
        if len(face_landmarks_list_2) == 0:
            logging.error('Frame: ' + str(frame_number) + ' - face not detected in target')
        return []

    points_source = []
    points_target = []
    total_faces = len(face_landmarks_list_1)
    for face_no in range(0, total_faces):
        points_1, points_2 = intersect(face_landmarks_list_1[face_no], face_landmarks_list_2[face_no])
        points_source.append(points_1)
        points_target.append(points_2)

    #visualizeFeatures(img, (values[:, 0], values[:, 1]))
    points_source = np.array(points_source).astype(np.int32)
    points_target = np.array(points_target).astype(np.int32)

    points_source = [listOfListToTuples(points.tolist()) for points in points_source]
    points_target = [listOfListToTuples(points.tolist()) for points in points_target]

    return face_landmarks_list_1, face_landmarks_list_2, points_source, points_target

'''Reference: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv/41075028'''
def landmark_detect_clahe2_helper(img):
    #Converting image to LAB Color model
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    #Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    #Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

def landmark_detect_clahe2(source_image, target_image, frame_no):
    si = landmark_detect_clahe2_helper(source_image)
    ti = landmark_detect_clahe2_helper(target_image)
    return landmark_detect(si, ti, frame_no)

def landmark_detect_clahe2_dual(source_image, target_image, frame_no):
    si = landmark_detect_clahe2_helper(source_image)
    ti = landmark_detect_clahe2_helper(target_image)
    return landmark_detect_dual(si, ti, frame_no)

def landmark_detect_clahe2_multi(source_image, target_image, frame_no):
    si = landmark_detect_clahe2_helper(source_image)
    ti = landmark_detect_clahe2_helper(target_image)
    return landmark_detect(si, ti, frame_no)

def landmark_detect_clahe(source_image, target_image, frame_no):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    source_gray = convert_BGR2Gray(source_image)
    target_gray = convert_BGR2Gray(target_image)

    cl1 = clahe.apply(source_gray)
    cl2 = clahe.apply(target_gray)

    return landmark_detect_dual(cl1, cl2, frame_no)

def intersect(face_landmarks_1, face_landmarks_2):
    points_1,points_2 = [],[]

    for key_1, value_1 in face_landmarks_1.items():
        if key_1 in face_landmarks_2:
            value_2 = face_landmarks_2[key_1]
            if len(value_1) == len(value_2):
                points_1.extend(value_1)
                points_2.extend(value_2)

    return points_1,points_2

