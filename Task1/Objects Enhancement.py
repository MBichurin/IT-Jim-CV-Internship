import cv2
import numpy as np

if __name__ == '__main__':
    vid_name = 'input_video.avi'
    vid = cv2.VideoCapture(vid_name)
    ret, blackframe = vid.read()
    while ret:
        # Get rid of black frame
        frame_bgr = blackframe[60:420, :]

        # Resize
        frame_bgr = cv2.resize(frame_bgr, (int(frame_bgr.shape[1] * 0.8), int(frame_bgr.shape[0] * 0.8)))

        # BGR 2 GRAY
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # BGR 2 HSV
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # BGR 2 Lab
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)

        # Adaptive Threshold Gauss of Gray
        frame_adaptGauss = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # image = 2 versions of a frame
        image = np.zeros((frame_gray.shape[0] * 2, frame_gray.shape[1]), dtype=np.uint8)
        image[:frame_gray.shape[0], :] = frame_gray[:, :]
        image[frame_gray.shape[0]:, :] = frame_adaptGauss

        cv2.imshow('Player', image)
        cv2.waitKey(30)

        ret, blackframe = vid.read()