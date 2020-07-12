import cv2
import numpy as np

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
    # 2-lvl contours to ignore inners
    contours, hierarchy = cv2.findContours(frm_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    frame_cnts = np.zeros(frm_edges.shape, dtype=np.uint8)

    # Top border of frames
    topBorder = np.zeros(frame_cnts.shape, dtype=np.uint8)
    topBorder[0, :] = 255

    i_cnt = 0
    while i_cnt >= 0 and len(contours) > 0:  # the contour exists
        the_cnt = contours[i_cnt]
        # Perimeter
        P = cv2.arcLength(the_cnt, False)

        if 100 < P < 1200: # Get rid of small and large objs
            # Put the contour on an image to check if it's on borders
            cnt_img = np.zeros(frame_cnts.shape, dtype=np.uint8)
            cnt_img = cv2.drawContours(cnt_img, [the_cnt], -1, (255, 255, 255), 1)

            if ~cv2.bitwise_and(cnt_img, topBorder).any(): # Cnt doesn't touch borders
                # Rotated Rectangle
                (x, y), (width, height), angle = cv2.minAreaRect(the_cnt)
                S_rect = width * height

                if width > height:
                    width, height = height, width

                if width > 0 and height / width < 1.5: # Not prolate obj
                    frame_cnts = cv2.fillPoly(frame_cnts, [the_cnt], (255, 255, 255))

        # Next contour
        i_cnt = hierarchy[0][i_cnt][0]

    # Erode
    frame_cnts = cv2.erode(frame_cnts, np.ones((21, 21), np.uint8))

    # Dilate
    for i in range(2):
        frame_cnts = cv2.dilate(frame_cnts, np.ones((7, 7), np.uint8))

    return frame_cnts

def approxFigures(mask):
    # 1-lvl contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(mask.shape, dtype=np.uint8)

    i_cnt = 0
    while i_cnt >= 0 and len(contours) > 0:  # cnt exists
        the_cnt = contours[i_cnt]

        # Convex Hull
        hull = cv2.convexHull(the_cnt)

        # Straight Bounding Rectangle
        x, y, width, height = cv2.boundingRect(hull)

        if width * height >= 385: # Get rid of small objs
            mask = cv2.fillPoly(mask, [hull], (255, 255, 255))

        # Next contour
        i_cnt = hierarchy[0][i_cnt][0]
    return mask

def findBlackObjs(frame_gray):
    # Black clr only
    black_mask = cv2.inRange(frame_gray, 0, 30)

    # Open
    black_mask = cv2.erode(black_mask, np.ones((7, 7), np.uint8))
    black_mask = cv2.dilate(black_mask, np.ones((7, 7), np.uint8))

    # 2-lvl contours to ignore inners
    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    black_mask = np.zeros(black_mask.shape, dtype=np.uint8)

    # Top border of frames
    topBorder = np.zeros(black_mask.shape, dtype=np.uint8)
    topBorder[0, :] = 255

    i_cnt = 0
    while i_cnt >= 0 and len(contours) > 0:  # the contour exists
        the_cnt = contours[i_cnt]

        # Area
        S = cv2.contourArea(the_cnt)

        if S > 3800:  # Get rid of small objs
            # Put the contour on an image to check if it's on borders
            cnt_img = np.zeros(black_mask.shape, dtype=np.uint8)
            cnt_img = cv2.drawContours(cnt_img, [the_cnt], -1, (255, 255, 255), 1)

            if ~cv2.bitwise_and(cnt_img, topBorder).any():  # Cnt doesn't touch borders
                # Convex Hull
                hull = cv2.convexHull(the_cnt)

                black_mask = cv2.fillPoly(black_mask, [hull], (255, 255, 255))

        # Next contour
        i_cnt = hierarchy[0][i_cnt][0]

    # Dilate
    for i in range(2):
        black_mask = cv2.dilate(black_mask, np.ones((3, 3), np.uint8))

    return black_mask

def detectionRects(mask):
    # 2-lvl contours to ignore inners
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(mask.shape, dtype=np.uint8)

    i_cnt = 0
    while i_cnt >= 0 and len(contours) > 0:  # the contour exists
        the_cnt = contours[i_cnt]

        x, y, width, height = cv2.boundingRect(the_cnt)
        cv2.rectangle(mask, (x, y), (x + width, y + height), (255, 255, 255), 2)

        # Next contour
        i_cnt = hierarchy[0][i_cnt][0]

    return mask

if __name__ == '__main__':
    vid_name = 'input_video.avi'
    vid = cv2.VideoCapture(vid_name)
    ret, blackframe = vid.read()

    key = None

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('output_video.avi', fourcc, 30, (int(blackframe.shape[1] * 0.8), int(blackframe.shape[0] * 0.8)), 0)

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

        # Approx figures
        mask = approxFigures(mask)

        # Black color
        black_mask = findBlackObjs(frame_gray)
        mask = cv2.bitwise_or(mask, black_mask)

        # Frame with mask
        frame_mask = np.copy(frame_bgr)
        frame_mask[mask == 255, :] = (255, 102, 102)

        # Rects of detections
        rects = detectionRects(mask)
        frame_det_rects = np.copy(frame_bgr)
        frame_det_rects[rects == 255, :] = (0, 255, 0)

        # Upd the main image
        image = np.zeros((frame_bgr.shape[0] * 2, frame_bgr.shape[1] * 2, frame_bgr.shape[2]), dtype=np.uint8)
        image[:frame_gray.shape[0], :frame_bgr.shape[1], :] = frame_bgr
        image[:frame_gray.shape[0], frame_bgr.shape[1]:, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image[frame_gray.shape[0]:, :frame_bgr.shape[1], :] = frame_mask
        image[frame_gray.shape[0]:, frame_bgr.shape[1]:, :] = frame_det_rects

        # Upd screen
        cv2.imshow('Objects detection', image)

        if key == ord(' '):
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(25)

        # Write frame in output video
        frame_out = np.zeros((int(blackframe.shape[0] * 0.8), int(blackframe.shape[1] * 0.8)), dtype='uint8')
        frame_out[int(60 * 0.8):int(420 * 0.8), :] = mask
        writer.write(frame_out)

        # Get the next frame
        ret, blackframe = vid.read()
    vid.release()
    writer.release()
    cv2.destroyAllWindows()