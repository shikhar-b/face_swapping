import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers

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


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
						 borderMode=cv2.BORDER_REFLECT_101)

	return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
	if point[0] < rect[0]:
		return False
	elif point[1] < rect[1]:
		return False
	elif point[0] > rect[2]:
		return False
	elif point[1] > rect[3]:
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
					if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
						ind.append(k)
					# Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
			if len(ind) == 3:
				delaunayTri.append((ind[0], ind[1], ind[2]))

		pt = []

	return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
	# Find bounding rectangle for each triangle
	r1 = cv2.boundingRect(np.float32([t1]))
	r2 = cv2.boundingRect(np.float32([t2]))

	# Offset points by left top corner of the respective rectangles
	t1Rect = []
	t2Rect = []
	t2RectInt = []

	for i in xrange(0, 3):
		t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
		t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
		t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

	# Get mask by filling triangle
	mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

	# Apply warpImage to small rectangular patches
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	# img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

	size = (r2[2], r2[3])

	img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

	img2Rect = img2Rect * mask

	# Copy triangular region of the rectangular patch to the output image
	img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
	(1.0, 1.0, 1.0) - mask)

	img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def visualizeFeatures(colorImg, p):
	p = np.array(p)
	showBGRimage(colorImg, (p[:, 0], p[:, 1]))

def listOfListToTuples(p):
	t_list = []
	for ent in p:
		t_list.append((ent[0], ent[1]))
	return t_list