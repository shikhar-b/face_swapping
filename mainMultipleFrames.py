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
from opticalFlow import *

SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
TARGET_PATH = 'datasets/Easy/MrRobot.mp4'

# SOURCE_PATH = 'datasets/Medium/LucianoRosso1.mp4'
# TARGET_PATH = 'datasets/Medium/LucianoRosso2.mp4'

# SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
# TARGET_PATH = 'datasets/Medium/LucianoRosso1.mp4'

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

			# if pos_frame > 20:
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
	fld1, fld2, points1, points2 = face_detection.landmark_detect_clahe2(tf, tf)
	# visualizeFeatures(sf, points1)
	visualizeFeatures(tf, points2)

#####################################################################################################

def getAllEncodings1(video):
	source_video_encodings = []
	numJitters = 3
	for frame in video:
		encoding = face_recognition.face_encodings(frame, num_jitters=numJitters)
		source_video_encodings.append(encoding[0]) # assuming only one face per video

	return source_video_encodings

def getClosestSourceFrame1(sourceEncodings, sourceVideo, targetFrame):
	numJitters = 3
	target_frame_encoding = face_recognition.face_encodings(targetFrame, num_jitters=numJitters)[0]
	distance = face_recognition.face_distance(sourceEncodings, target_frame_encoding)	
	sf = sourceVideo[np.argmin(distance)]
	return sf

#####################################################################################################

def getFrameFeatures(frame):
	f = face_detection.landmark_detect_clahe2_helper(frame)
	landmarks = face_recognition.face_landmarks(f)
	# visualizeFeatures(frame, np.asarray(landmarks[0]['left_eye'][0]).astype(np.int32)[:, None].T)

	leftEyeLoc = np.asarray(landmarks[0]['left_eye'][0]).astype(np.int32)[:, None].T.astype(np.float32)
	rightEyeLoc = np.asarray(landmarks[0]['right_eye'][3]).astype(np.int32)[:, None].T.astype(np.float32)
	noseTipLoc = np.asarray(landmarks[0]['nose_tip'][2]).astype(np.int32)[:, None].T.astype(np.float32)

	feature = np.abs(np.sum(np.square(noseTipLoc - leftEyeLoc))) / np.abs(np.sum(np.square(noseTipLoc - rightEyeLoc)))
	return feature

def getFeatureDistance(f1, f2):
	return np.abs(f1 - f2)

def getAllEncodings(video):
	source_video_encodings = []
	for frame in video:
		features = getFrameFeatures(frame)
		source_video_encodings.append(features)

	return source_video_encodings

def getClosestSourceFrame(sourceEncodings, sourceVideo, targetFrame):
	targetFeatures = getFrameFeatures(targetFrame)

	minDis = 999999999
	minIndex = 0
	for i, sf in enumerate(sourceEncodings):
		d = getFeatureDistance(sf, targetFeatures)

		if d < minDis:
			minDis = d
			minIndex = i

	return sourceVideo[minIndex]

#####################################################################################################

if __name__ == "__main__":

	source_video = loadVideo(SOURCE_PATH)
	target_video = loadVideo(TARGET_PATH)

	print ('Videos loaded')
	print ('Starting source video encoding')
	source_video_encodings = getAllEncodings(source_video)
	print ('Source video encodings found')

	print ('Starting to make target video')
	output_video = []
	points1 = []
	for frameNum, target_frame in enumerate(target_video):
		print ('Processing target frame # ' + str(frameNum))

		if frameNum % 4 == 0:
			try:
				sf = getClosestSourceFrame(source_video_encodings, source_video, target_frame)
			except:
				continue

			tf = target_frame
			img1Warped = np.copy(tf)

			#STEP 1: Landmark Detection
			try:
				fld1, fld2, points1 , points2 = face_detection.landmark_detect_clahe2(sf, tf)
			except KeyboardInterrupt:
				sys.exit()
			except:
				if len(points1) == 0:
					continue
			if empty_points(points1, points2, 1): continue

			# STEP 2: Convex Hull
			try:
				hull1, hull2 = convex_hull_internal_points(points1, points2, fld1, fld2)
			except KeyboardInterrupt:
				sys.exit()
			except:
				print (traceback.format_exc())
				continue
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
			prev_target_frame = target_frame
		else:
			try:
				output, points2 = doOpticalFlow(output, points2, target_frame, prev_target_frame)
				output_video.append(output)
				prev_target_frame = target_frame
			except KeyboardInterrupt:
				sys.exit()
			except:
				print (traceback.format_exc())
				continue

	saveVideo(output_video)
