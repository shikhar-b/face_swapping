import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import traceback

from helpers import *
import face_detection

SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
TARGET_PATH = 'datasets/Easy/MrRobot.mp4'

# SOURCE_PATH = 'datasets/Medium/LucianoRosso1.mp4'
# TARGET_PATH = 'datasets/Medium/LucianoRosso2.mp4'

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
	out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))


	try:
		while True:
			flag_source, source_frame = cap_source.read()
			flag_target, target_frame = cap_target.read()
			img1Warped = np.copy(target_frame);

			if flag_source and flag_target:
				pos_frame = cap_source.get(cv2.CAP_PROP_POS_FRAMES)
				print ''
				print pos_frame
				if pos_frame == 1 or True:
					#showBGRimage(source_frame)
					#showBGRimage(target_frame)
					points1 = face_detection.landmark_detect(source_frame)
					points2 = face_detection.landmark_detect(target_frame)

					print (len(points1))
					print (len(points2))

					if len(points1) == 0 or len(points2) == 0:
						continue

					# visualizeFeatures(source_frame, points1)
					# visualizeFeatures(target_frame, points2)

					# Find convex hull
					hull1 = []
					hull2 = []

					hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
					# pdb.set_trace()

					for i in xrange(0, len(hullIndex)):
						hull1.append(points1[int(hullIndex[i])])
						hull2.append(points2[int(hullIndex[i])])
					# visualizeFeatures(source_frame, hull1)
					# visualizeFeatures(target_frame, hull2)

					# Find delanauy traingulation for convex hull points
					sizeImg2 = target_frame.shape
					rect = (0, 0, sizeImg2[1], sizeImg2[0])

					dt = calculateDelaunayTriangles(rect, hull2, target_frame)

					if len(dt) == 0:
						print 'No delaunay'
						continue

					# Apply affine transformation to Delaunay triangles
					for i in xrange(0, len(dt)):
						t1 = []
						t2 = []

						# get points for img1, img2 corresponding to the triangles
						for j in xrange(0, 3):
							t1.append(hull1[dt[i][j]])
							t2.append(hull2[dt[i][j]])

						warpTriangle(source_frame, img1Warped, t1, t2)

					# Calculate Mask
					hull8U = []
					for i in xrange(0, len(hull2)):
						hull8U.append((hull2[i][0], hull2[i][1]))

					mask = np.zeros(target_frame.shape, dtype=target_frame.dtype)

					cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

					r = cv2.boundingRect(np.float32([hull2]))

					center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

					# Clone seamlessly.
					output = cv2.seamlessClone(np.uint8(img1Warped), target_frame, mask, center, cv2.NORMAL_CLONE)
					# showBGRimage(output)
					# cv2.imshow("Face Swapped", output)
					out.write(output)
				else:
					break
					print str(pos_frame) + " frames"
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

	cv2.destroyAllWindows()
	cap_source.release()
	cap_target.release()
	out.release()