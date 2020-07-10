import cv2
import numpy as np

def myEdgeSearch(frame_gray):
    # Adaptive Threshold Gauss of Gray
    frame_adaptGauss = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Inverse
    frame_adaptGauss = 255 - frame_adaptGauss

    # Opening
    frame_open = cv2.erode(frame_adaptGauss, np.ones((3, 3), np.uint8), 1)
    frame_open = cv2.dilate(frame_open, np.ones((3, 3), np.uint8), 1)

    return frame_open

if __name__ == '__main__':
    vid_name = 'input_video.avi'
    vid = cv2.VideoCapture(vid_name)
    ret, blackframe = vid.read()
    while ret:
        # Get rid of black frame
        frame_bgr = blackframe[60:420, :]

        # Resize
        frame_bgr = cv2.resize(frame_bgr, (int(frame_bgr.shape[1] * 0.8), int(frame_bgr.shape[0] * 0.8)))

        # Converts
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)



        # CannyV
        frame_cannyV = cv2.Canny(frame_hsv[:, :, 2], 35, 45)




        # image = 2 versions of a frame
        image = np.zeros((frame_gray.shape[0] * 2, frame_gray.shape[1]), dtype=np.uint8)

        image[:frame_gray.shape[0], :] = frame_gray

        image[frame_gray.shape[0]:, :] = frame_cannyV


        cv2.imshow('Player', image)
        cv2.waitKey(30)

        ret, blackframe = vid.read()