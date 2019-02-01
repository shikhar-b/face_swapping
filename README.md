# face_swapping
CIS 581 final project 

Requirements:

Python 2.7,
OpenCV 3,
Numpy,
Scipy,
face_recognition

Instructions:

"Source" - The face to be replaced. This will be the face that will be shown in the resulting video.
"Target" - The body into which the a new face will come. The body and surroundings of target video will be kept in the resulting video and the face will be of the source video.

Preserving source emotion:

	Run main.py after setting SOURCE_PATH for source video and TARGET_PATH for target video.
	Set frame rate for swap, for other frames we use optical flow for estimation.

	For Multiple Frame Video Matching, run mainMultipleFrames.py after setting SOURCE_PATH and TARGET_PATH inside it.

	For Single Video Multiple Face Swap, run main2Face.py after setting the same video in both SOURCE_PATH and TARGET_PATH.

Preserving target emotion:

	Run main_target_emotion.py after setting SOURCE_PATH for source video and TARGET_PATH for target video.
	Set frame rate for swap, for other frames we use optical flow for estimation.

	For Multiple Frame Video Matching, run mainMultipleFramesTargetEmotion.py after setting SOURCE_PATH and TARGET_PATH inside it.

	For Single Video Multiple Face Swap, run main2Face_target_emotion.py after setting the same video in both SOURCE_PATH and TARGET_PATH.


Demo Videos:
https://www.youtube.com/watch?v=PM9SF4uI2k0
