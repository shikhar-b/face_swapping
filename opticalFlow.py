import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import traceback
import logging, time
from skimage.transform import SimilarityTransform, matrix_transform
from helpers import *

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

def doOpticalFlow(prevOutput, targetPoints, target_frame, prev_target_frame):

	# visualizeFeatures(prevOutput, targetPoints)
	# visualizeFeatures(prevOutput, targetHull)

	# showBGRimage(prev_target_frame)
	# showBGRimage(target_frame)
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
		homography = transform.params
		# newBbox[i] = transformation(homography, bbox[i])

		# get hull from these new points in the new target frame

		# make a convex fill mask inside this hull

		# for each point in the new image, copy from the old image/prevOutput (use transformation/float points..)

		# use the above mask to only keep the masked part of the above copy

		# do seamless cloning or may not be required..


		pdb.set_trace()


	return newOutput, listOfListToTuples(good_new.tolist())