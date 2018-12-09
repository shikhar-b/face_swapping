import cv2,numpy as np
from helpers import *

def cloning(img1Warped, target_frame, hull_target):

    # Calculate Mask
    hull_target = np.squeeze(cv2.convexHull(np.array(hull_target).astype(np.int32), returnPoints=True))
    hull_target_tuple = listOfListToTuples(hull_target)

    mask = np.zeros(target_frame.shape, dtype=target_frame.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull_target_tuple), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull_target]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), target_frame, mask, center, cv2.NORMAL_CLONE)

    return output