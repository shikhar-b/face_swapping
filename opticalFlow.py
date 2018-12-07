import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import traceback
import logging, time
from helpers import *

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
					maxLevel = 2,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


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


	pdb.set_trace()