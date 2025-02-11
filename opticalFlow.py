import cv2, numpy as np
import pdb, logging
from skimage.transform import SimilarityTransform, matrix_transform
from helpers import *
from convex_hull import *
from triangulation import triangulation
from warping import warping
from face_detection import *

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
					maxLevel = 2,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def transformation(H, x):
	newBbox_temp = np.matmul(H, np.hstack((x, np.ones((x.shape[0], 1))))[:, :, None])
	newBbox_temp = np.squeeze(newBbox_temp)
	newBbox_temp = newBbox_temp[:, 0:2] / newBbox_temp[:, 2][:, None]
	newBbox_temp = np.squeeze(newBbox_temp)
	return newBbox_temp

def doOpticalFlow(prevOutput, targetPoints, target_frame, prev_target_frame, frame_no):
	p0 = np.asarray(targetPoints).astype(np.float32)[:, :, None]
	p0 = np.transpose(p0, (0, 2, 1))
	old_gray = cv2.cvtColor(prev_target_frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	newOutput = np.copy(prevOutput)

	transform = SimilarityTransform()
	if transform.estimate(good_old, good_new):
		newOutput = transform_image(good_old, good_new, prevOutput, target_frame, frame_no)
		# make a convex fill mask inside this hull
		# for each point in the new image, copy from the old image/prevOutput (use transformation/float points..)
		# use the above mask to only keep the masked part of the above copy
		# seamless cloning will not be required.

	return newOutput, listOfListToTuples(good_new.tolist())

def transform_image(points1, points2, img1, img2, frame_no):
	img1Warped = np.copy(img2)
	#face_landmarks_dict_1, face_landmarks_dict_2, _, _ = landmark_detect_clahe2(img1,img2,frame_no)
	#visualizeFeatures(img2, points2)
	#hull1, hull2 = convex_hull_all_internal_points(points1.tolist(), points2.tolist(), face_landmarks_dict_1, face_landmarks_dict_2)
	hull1, hull2 = convex_hull(points1.tolist(), points2.tolist())
	#visualizeFeatures(img2, hull2)
	hull1 = np.array(hull1).astype(np.float32)
	hull2 = np.array(hull2).astype(np.float32)
	if empty_points(hull1, hull2, 2, frame_no): return img2

	hull2 = np.asarray(hull2)
	hull2[:, 0] = np.clip(hull2[:, 0], 0, img2.shape[1] - 1)
	hull2[:, 1] = np.clip(hull2[:, 1], 0, img2.shape[0] - 1)
	hull2 = listOfListToTuples(hull2.astype(np.float32).tolist())

	dt = triangulation(img2, hull2)
	if len(dt) == 0:
		logging.error('Frame: '+ frame_no+ ' delaunay triangulation empty')
		return img2

	warping(dt, hull1, hull2, img1, img1Warped)

	return img1Warped