import cv2, numpy as np

def convex_hull(points1, points2):
    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    # pdb.set_trace()

    for i in xrange(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    return hull1, hull2