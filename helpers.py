import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb, logging

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

# Check if a point is inside a rectangle
def rectContains(rect, point):
	if point[0] < rect[0]:
		return False
	elif point[1] < rect[1]:
		return False
	elif point[0] > rect[0] + rect[2]:
		return False
	elif point[1] > rect[1] + rect[3]:
		return False
	return True

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color = (255, 255, 255)) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def visualizeFeatures(colorImg, p):
	p = np.array(p)
	showBGRimage(colorImg, (p[:, 0], p[:, 1]))

def listOfListToTuples(p):
	t_list = []
	for ent in p:
		s = []
		for i in range(0, len(ent)):
			s.append(ent[i])
		t_list.append(tuple(s))

	return t_list

def empty_points(points1, points2, step):
	if len(points1) == 0 or len(points2) == 0:
		if len(points1) == 0 and len(points2) == 0:
			logging.info(str(step) + ' points1 and points2 empty')
		elif len(points1) == 0:
			logging.info(str(step) + 'points1 empty')
		else:
			logging.info(str(step) + 'points2 empty')
		return True

	return False

def convert_BGR2Gray(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img