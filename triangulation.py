import cv2, numpy as np
from helpers import *

def triangulation(target_frame, hull2):
    # Find delanauy traingulation for convex hull points
    sizeImg2 = target_frame.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2, target_frame)

    return dt


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points, img):
	# create subdiv
	subdiv = cv2.Subdiv2D(rect);

	# Insert points into subdiv
	for p in points:
		subdiv.insert(p)

	# imgToShow = np.copy(img)
	# draw_delaunay(imgToShow, subdiv)
	# showBGRimage(imgToShow)

	triangleList = subdiv.getTriangleList();

	delaunayTri = []

	pt = []

	for t in triangleList:
		pt.append((t[0], t[1]))
		pt.append((t[2], t[3]))
		pt.append((t[4], t[5]))

		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
			ind = []
			# Get face-points (from 68 face detector) by coordinates
			for j in xrange(0, 3):
				for k in xrange(0, len(points)):
					#if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
					ind.append(k)
					# Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
			if len(ind) == 3:
				delaunayTri.append((ind[0], ind[1], ind[2]))


		pt = []

	return delaunayTri
