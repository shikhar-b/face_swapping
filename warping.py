import numpy as np
import cv2
from helpers import *

def warping(dt, hull1, hull2, source_frame, img1Warped):
	# Apply affine transformation to Delaunay triangles
    for i in xrange(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in xrange(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(source_frame, img1Warped, t1, t2)

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
	# Find bounding rectangle for each triangle
	r1 = cv2.boundingRect(np.float32([t1]))
	r2 = cv2.boundingRect(np.float32([t2]))
	# pdb.set_trace()
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
	print (img2Rect.size)
	img2Rect = img2Rect * mask

	# Copy triangular region of the rectangular patch to the output image
	img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
	(1.0, 1.0, 1.0) - mask)

	img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

	showBGRimage(img2)
# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
						 borderMode=cv2.BORDER_REFLECT_101)

	return dst