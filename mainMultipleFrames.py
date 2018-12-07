import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import traceback
import logging, time
from helpers import *
import face_detection
import face_recognition
from convex_hull import *
from triangulation import *
from warping import *
from cloning import *

SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
TARGET_PATH = 'datasets/Easy/MrRobot.mp4'

# SOURCE_PATH = 'datasets/Medium/LucianoRosso1.mp4'
# TARGET_PATH = 'datasets/Medium/LucianoRosso2.mp4'

SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
TARGET_PATH = 'datasets/Medium/LucianoRosso1.mp4'

# SOURCE_PATH = 'datasets/Hard/Joker.mp4'
# TARGET_PATH = 'datasets/Hard/LeonardoDiCaprio.mp4'

def loadVideo(path):
	video = []
	cap_source = cv2.VideoCapture(path)
	videoSpecific1(cap_source, path)
	start = time.time()
	try:
		while True:
			flag_source, source_frame = cap_source.read()

			if flag_source:
				pos_frame = cap_source.get(cv2.CAP_PROP_POS_FRAMES)
				video.append(source_frame)

			# if pos_frame > 10:
			# 	break

			if cv2.waitKey(10) == 27 or cap_source.get(cv2.CAP_PROP_POS_FRAMES) == cap_source.get(cv2.CAP_PROP_FRAME_COUNT):
				break

	except Exception as e:
		print (traceback.format_exc())
	print ('time taken :' + str(time.time() - start))
	cv2.destroyAllWindows()
	cap_source.release()
	return video

def saveVideo(video, path = 'outputMultiple.avi'):
	if len(video) > 0:
		frame_height, frame_width, channels = video[0].shape
		out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame_width, frame_height))
		for frame in video:
			out.write(frame)
		out.release()
	else:
		print ('No video to write')

def showFrame(video, frameNum):
	tf = video[frameNum]
	points1, points2 = face_detection.landmark_detect_clahe(tf, tf)
	# visualizeFeatures(sf, points1)
	visualizeFeatures(tf, points2)

if __name__ == "__main__":

	source_video = loadVideo(SOURCE_PATH)
	target_video = loadVideo(TARGET_PATH)

	# showFrame(target_video, 100)
	# exit()

	print ('Videos loaded')
	print ('Starting source video encoding')
	source_video_encodings = []
	numJitters = 3
	for frame in source_video:
		encoding = face_recognition.face_encodings(frame, num_jitters=numJitters)
		source_video_encodings.append(encoding[0]) # assuming only one face per video
	print ('Source video encodings found')

	print ('Starting to make target video')
	output_video = []
	points1 = []
	for frameNum, target_frame in enumerate(target_video):
		print ('Processing target frame # ' + str(frameNum))
		target_frame_encoding = face_recognition.face_encodings(target_frame, num_jitters=numJitters)[0]
		distance = face_recognition.face_distance(source_video_encodings, target_frame_encoding)
		
		sf = source_video[np.argmin(distance)]
		tf = target_frame
		img1Warped = np.copy(tf)

		#STEP 1: Landmark Detection
		# points1, points2 = face_detection.landmark_detect_clahe(sf, tf)
		try:
			fld1, fld2, points1 , points2 = face_detection.landmark_detect_clahe(sf, tf)
		except:
			if len(points1) == 0:
				continue
		if empty_points(points1, points2, 1): continue
		#visualizeFeatures(sf, points1)
		#visualizeFeatures(tf, points2)

		# STEP 2: Convex Hull
		# hull1, hull2 = convex_hull(points1, points2)
		hull1, hull2 = convex_hull_internal_points(points1, points2, fld1, fld2)
		# visualizeFeatures(sf, hull1)
		if empty_points(hull1, hull2, 2): continue

		hull2 = np.asarray(hull2)
		hull2[:, 0] = np.clip(hull2[:, 0], 0, target_frame.shape[1] - 1)
		hull2[:, 1] = np.clip(hull2[:, 1], 0, target_frame.shape[0] - 1)
		hull2 = listOfListToTuples(hull2.astype(np.int32).tolist())

		# STEP 3: Triangulation
		dt = triangulation(tf, hull2)
		if len(dt) == 0:
			logging.info('delaunay triangulation empty')
			continue

		# STEP 4: Warping
		warping(dt, hull1, hull2, sf, img1Warped)

		# STEP 5: Cloning
		output = cloning(img1Warped, tf, hull2)

		output_video.append(output)

	saveVideo(output_video)
