import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import traceback

from helpers import *
import face_detection
SOURCE_PATH = 'datasets/Easy/FrankUnderwood.mp4'
TARGET_PATH = 'datasets/Easy/MrRobot.mp4'
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

			if flag_source and flag_target:
				pos_frame = cap_source.get(cv2.CAP_PROP_POS_FRAMES)
				if pos_frame == 1:
					showBGRimage(source_frame)
					#showBGRimage(target_frame)
					face_detection.face_detect(source_frame)
					#out.write(input_frame)
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