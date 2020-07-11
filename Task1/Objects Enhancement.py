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

    iter = 0

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

        # Blur
        frame_cannyV = cv2.blur(frame_cannyV, (5, 5))
        frame_cannyV = cv2.blur(frame_cannyV, (3, 3))
        frame_cannyV = cv2.blur(frame_cannyV, (3, 3))

        # Open
        frame_cannyV = cv2.morphologyEx(frame_cannyV, cv2.MORPH_OPEN, np.ones(5))

        # Close
        frame_cannyV = cv2.morphologyEx(frame_cannyV, cv2.MORPH_CLOSE, np.ones(5))

        # Contours
        contours, hierarchy = cv2.findContours(frame_cannyV, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


        # image = 2 versions of a frame
        image = np.zeros((frame_gray.shape[0] * 2, frame_gray.shape[1], 3), dtype=np.uint8)


        i_cnt = 0
        specialCnt = np.ones(frame_bgr.shape)
        while i_cnt >= 0: # the contour exists
            # Fill the main image

            fst_child = hierarchy[0][i_cnt][2]
            if fst_child < 0 or (fst_child >= 0 and hierarchy[0][fst_child][0] >= 0): # no child or > 1 child
                P = cv2.arcLength(contours[i_cnt], False)
                if 100 < P < 1200:
                    image[:frame_gray.shape[0], :, :] = cv2.drawContours(specialCnt, contours, i_cnt, (255, 255, 0), 3)
            else:
                P = cv2.arcLength(contours[fst_child], False)
                if 100 < P < 1200:
                    image[:frame_gray.shape[0], :, :] = cv2.drawContours(specialCnt, contours, fst_child, (0, 255, 0), 3)
                else:
                    P = cv2.arcLength(contours[i_cnt], False)
                    if 100 < P < 1200:
                        image[:frame_gray.shape[0], :, :] = cv2.drawContours(specialCnt, contours, i_cnt, (255, 255, 0), 3)

            image[frame_gray.shape[0]:, :, 0] = frame_cannyV

            # Next contour
            i_cnt = hierarchy[0][i_cnt][0]

            # Build a window
            cv2.imshow('Player', image)

        cv2.waitKey(30)

        # Get the next frame
        ret, blackframe = vid.read()

        iter += 1
    vid.release()