import cv2
import numpy as np
import pdb, sys
import matplotlib.pyplot as plt
import traceback
import logging, time
from helpers import *
import face_detection
from convex_hull import convex_hull, convex_hull_internal_points
from triangulation import triangulation
from warping import warping
from cloning import cloning
from opticalFlow import *

SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
TARGET_PATH = 'datasets/Easy/MrRobot.mp4'

#SOURCE_PATH = 'datasets/Medium/LucianoRosso1.mp4'
#TARGET_PATH = 'datasets/Medium/LucianoRosso2.mp4'

# SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
# TARGET_PATH= 'datasets/Medium/LucianoRosso1.mp4'

# SOURCE_PATH = 'datasets/Hard/Joker.mp4'
# TARGET_PATH = 'datasets/Hard/LeonardoDiCaprio.mp4'

if __name__ == "__main__":
	cap_source = cv2.VideoCapture(SOURCE_PATH)
	videoSpecific1(cap_source, SOURCE_PATH)

	cap_target = cv2.VideoCapture(TARGET_PATH)
	videoSpecific1(cap_target, TARGET_PATH)

	pos_frame = cap_target.get(cv2.CAP_PROP_POS_FRAMES)
	frame_width = int(cap_target.get(3))
	frame_height = int(cap_target.get(4))
	out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame_width, frame_height))
	limit = 1000
	start = time.time()
	points1 = []
	jump = 137
	try:
		while True:
			flag_source, source_frame = cap_source.read()
			flag_target, target_frame = cap_target.read()
			img1Warped = np.copy(target_frame)

			if flag_source and flag_target:
				pos_frame = cap_target.get(cv2.CAP_PROP_POS_FRAMES)
				# while pos_frame<jump:
				# 	flag_target, target_frame = cap_target.read()
				# 	pos_frame = cap_target.get(cv2.CAP_PROP_POS_FRAMES)
				# 	img1Warped = np.copy(target_frame)
				print pos_frame
				if (pos_frame-1) % 3 == 0 :

					#STEP 1: Landmark Detection
					try:
						face_landmarks_dict_1, face_landmarks_dict_2, points1, points2 = face_detection.landmark_detect_clahe2(source_frame, target_frame)
					except KeyboardInterrupt:
						sys.exit()
					except:
						if len(points1) == 0:
							continue

					if empty_points(points1, points2, 1): continue

					# visualizeFeatures(source_frame, points1)
					# visualizeFeatures(target_frame, points2)

					# STEP 2: Convex Hull


					hull1, hull2 = convex_hull_internal_points(points1, points2, face_landmarks_dict_1,face_landmarks_dict_2)
					# visualizeFeatures(source_frame, hull1)
					if empty_points(hull1, hull2, 2): continue

					hull2 = np.asarray(hull2)
					hull2[:, 0] = np.clip(hull2[:, 0], 0, target_frame.shape[1] - 1)
					hull2[:, 1] = np.clip(hull2[:, 1], 0, target_frame.shape[0] - 1)
					hull2 = listOfListToTuples(hull2.astype(np.int32).tolist())


					# STEP 3: Triangulation
					dt = triangulation(target_frame, hull2)
					if len(dt) == 0:
						logging.info('delaunay triangulation empty')
						continue
					print('len dt :' +str(len(dt)))
					# STEP 4: Warping
					warping(dt, hull1, hull2, source_frame, img1Warped)

					# STEP 5: Cloning
					output = cloning(img1Warped, target_frame, hull2)

					# cv2.imshow("Face Swapped", output)
					out.write(output)
					prev_target_frame = target_frame

					if pos_frame == limit:
						print ('time taken :' + str(time.time() - start))
						cv2.destroyAllWindows()
						cap_source.release()
						cap_target.release()
						out.release()
						exit()
				else:
					try:
						output, points2 = doOpticalFlow(output, points2, target_frame, prev_target_frame)
						out.write(output)
						prev_target_frame = target_frame
					except KeyboardInterrupt:
						sys.exit()
					except:
						print (traceback.format_exc())
						exit()
						continue
			else:
				if flag_source:
					videoSpecific2(cap_source, pos_frame)
				if flag_target:
					videoSpecific2(cap_target, pos_frame)

			if cv2.waitKey(10) == 27 or cap_target.get(cv2.CAP_PROP_POS_FRAMES) == cap_target.get(cv2.CAP_PROP_FRAME_COUNT)\
					or cap_source.get(cv2.CAP_PROP_POS_FRAMES) == cap_source.get(cv2.CAP_PROP_FRAME_COUNT):
				break

	except Exception as e:
		print (traceback.format_exc())

	print ('time taken :' + str(time.time() - start))
	cv2.destroyAllWindows()
	cap_source.release()
	cap_target.release()
	out.release()