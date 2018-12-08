import cv2, numpy as np, pdb

def convex_hull(points1, points2):
    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2).astype(np.int32), returnPoints=False)
    # pdb.set_trace()

    for i in xrange(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    return hull1, hull2

def add_point(feature_name, face_landmarks_dict_1, face_landmarks_dict_2, hull1, hull2):
    if feature_name in face_landmarks_dict_1 and feature_name in face_landmarks_dict_2:
        if len(face_landmarks_dict_1[feature_name])!=0 and len(face_landmarks_dict_2[feature_name])!=0:
            hull1.append(face_landmarks_dict_1[feature_name][0])
            hull2.append(face_landmarks_dict_2[feature_name][0])


def convex_hull_internal_points(points1, points2, face_landmarks_dict_1,face_landmarks_dict_2):
    # Find convex hull
    hull1 = []
    hull2 = []

    # pdb.set_trace()
    print (len(points2))
    hullIndex = cv2.convexHull(np.array(points2).astype(np.int32), returnPoints=False)

    for i in xrange(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])
    features = ['left_eye', 'right_eye', 'nose_bridge', 'nose_tip', 'top_lip']
    for feature in features:
        add_point(feature, face_landmarks_dict_1, face_landmarks_dict_2, hull1, hull2)
    return hull1, hull2