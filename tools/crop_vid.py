import cv2
from helpers import *
vidcap = cv2.VideoCapture('videoplayback.mp4')
pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame_width, frame_height))

success,image = vidcap.read()
start = 3200 #Start frame here
end = 3400 #End frame here
while pos_frame<start:
    flag_target, frame = vidcap.read()
    pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
showBGRimage(frame)
while pos_frame<end:
    flag_target, frame = vidcap.read()
    out.write(frame)
    pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)

cv2.destroyAllWindows()
vidcap.release()
out.release()