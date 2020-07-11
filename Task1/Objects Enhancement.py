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

def cannyEdgeSearch(frame_hsv):
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
    return frame_cannyV

def getMask(frm_edges):
    # Contours
    contours, hierarchy = cv2.findContours(frm_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    i_cnt = 0

    frame_cnts = np.zeros(frm_edges.shape, dtype=np.uint8)

    # Top border of frames
    topBorder = np.zeros(frame_cnts.shape, dtype=np.uint8)
    topBorder[0, :] = 255

    while i_cnt >= 0:  # the contour exists
        the_cnt = contours[i_cnt]
        # Perimeter
        P = cv2.arcLength(the_cnt, False)
        S = cv2.contourArea(the_cnt)

        # Enclosing Circle
        (x, y), radius = cv2.minEnclosingCircle(the_cnt)
        S_circle = np.pi * (radius ** 2)

        # Rotated Rectangle
        (x, y), (width, height), angle = cv2.minAreaRect(the_cnt)
        S_rect = width * height
        if width > height:
            width, height = height, width

        # Put the contour on an image to check if it's on borders
        cnt_img = np.zeros(frame_cnts.shape, dtype=np.uint8)
        cnt_img = cv2.drawContours(cnt_img, [the_cnt], -1, (255, 255, 255), 1)

        if 100 < P < 1200 and height / width < 1.5 and ~cv2.bitwise_and(cnt_img, topBorder).any():
            # frame_cnts = cv2.drawContours(specialCnt, the_cnt, -1, (255, 255, 255), 1)
            frame_cnts = cv2.fillPoly(frame_cnts, [the_cnt], (255, 255, 255))

        # Next contour
        i_cnt = hierarchy[0][i_cnt][0]
    return frame_cnts

def approxFigures(mask):
    # Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    i_cnt = 0

    mask = np.zeros(mask.shape, dtype=np.uint8)

    while i_cnt >= 0 and len(contours) > 0:  # the contour exists
        the_cnt = contours[i_cnt]

        hull = cv2.convexHull(the_cnt)

        mask = cv2.fillPoly(mask, [hull], (255, 255, 255))

        # Next contour
        i_cnt = hierarchy[0][i_cnt][0]
    return mask

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

        # Find edges
        frm_edges = cannyEdgeSearch(frame_hsv)

        # Get mask
        mask = getMask(frm_edges)

        # Erode (maybe (7,7) 3*times is better)
        mask = cv2.erode(mask, np.ones((21, 21), np.uint8))

        # Dilate
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))

        # Approx figures
        mask = approxFigures(mask)

        # Upd the main image
        image = np.zeros((frame_gray.shape[0] * 2, frame_gray.shape[1]), dtype=np.uint8)
        image[:frame_gray.shape[0], :] = mask
        image[frame_gray.shape[0]:, :] = cv2.bitwise_or(frame_gray, mask)

        # Build a window
        cv2.imshow('Player', image)

        cv2.waitKey(30)

        # Get the next frame
        ret, blackframe = vid.read()

        iter += 1
    vid.release()