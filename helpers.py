import numpy as np
import cv2
import matplotlib.pyplot as plt

def videoSpecific1(cap, videoPath):
	while not cap.isOpened():
		cap = cv2.VideoCapture(videoPath)
		cv2.waitKey(1000)
		print "Wait for the header"

def videoSpecific2(cap, pos_frame):
	# The next frame is not ready, so we try to read it again
	cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
	print "frame is not ready"
	# It is better to wait for a while for the next frame to be ready
	cv2.waitKey(1000)

def showRGBimage(rgbImg, points=None):
	plt.imshow(rgbImg)

	if points != None:
		sx, sy = points
		plt.scatter(sx, sy)

	plt.show()

def showBGRimage(img, points=None):
	imgCopy = np.copy(img)
	rgbImg = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB)
	plt.imshow(rgbImg)

	if points != None:
		sx, sy = points
		plt.scatter(sx, sy)

	plt.show()

def showGrayImage(img):
	plt.imshow(img, cmap='gray')
	plt.show()