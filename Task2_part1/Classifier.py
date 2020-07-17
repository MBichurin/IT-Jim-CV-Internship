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

def classifyContour(cnt):
    # Perimeter and Area
    P = cv2.arcLength(cnt, True)
    S = cv2.contourArea(cnt)
    # Approximation and its area
    approx = cv2.approxPolyDP(cnt, 0.07 * P, True)
    S_approx = cv2.contourArea(approx)

    # Put the contour on an image
    cnt_img = np.zeros(black_mask.shape, dtype=np.uint8)
    cv2.drawContours(cnt_img, [cnt], -1, (255, 255, 255), 1)

    # Rotated rectangle
    (x, y), (w, h), angle = cv2.minAreaRect(cnt)
    # Min enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    # Borders of frames
    borders = np.zeros(black_mask.shape, dtype=np.uint8)
    borders[0, :] = 255
    borders[:, 0] = 255
    borders[borders.shape[0] - 1, :] = 255
    borders[:, borders.shape[1] - 1] = 255

    if S < 500 or cv2.bitwise_and(cnt_img, borders).any():
        # ??? <== the object is too small to be classified or it touches frame borders
        return -1
    elif w * h / S >= 1.4:
        # triangle <== the rotated rectangle's area is much bigger than the object's area
        return 0
    elif np.pi * radius * radius / S <= 1.365:
        # circle <== the enclosing circle's area is almost the same as the object's area
        return 1
    elif w * h / S <= 1.09:
        # rectangle <== the rotated rectangle's area is almost the same as the object's area
        return 2
    elif w * h / S_approx >= 1.8 and approx.shape[0] == 3:
        # triangle <== the approximation is a triangle and it's much smaller than the rotated rectangle's area
        return 0
    else:
        # more rough approximation
        approx = cv2.approxPolyDP(cnt, 0.2 * P, True)
        if approx.shape[0] == 3:
            # triangle <== the approximation is a triangle
            return 0
        else:
            # less rough approximation
            approx = cv2.approxPolyDP(cnt, 0.03 * P, True)
            (x, y), (w, h), angle = cv2.minAreaRect(approx)
            if w * h / cv2.contourArea(approx) <= 1.2 and approx.shape[0] == 4:
                # rectangle <== the approximation is a quadrilateral and the rotated rectangle's area is almost the same as the object's area
                return 2
            else:
                # ??? I can't divide the rest of objs because of the chosen detection strategy:(
                return -1

def classifyObjs(mask):
    # Only outer contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    classes = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Triangle, circle and rectangle colors
    colors = ((0, 255, 0), (153, 0, 153), (255, 102, 102))

    # List of lists of centers' coordinates: [[<triangles' centers>], [<circles' centers>], [<rects' centers>]]
    Centers = [[], [], []]

    i_cnt = 0
    while i_cnt >= 0 and len(contours) > 0:  # the contour exists
        the_cnt = contours[i_cnt]

        # Classify the contour (0 = triangle, 1 = circle, 2 = rectangle)
        cnt_class = classifyContour(the_cnt)

        if cnt_class != -1:
            cv2.drawContours(classes, [the_cnt], -1, colors[cnt_class], 3)
            M = cv2.moments(the_cnt)
            Centers[cnt_class].append([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])

        # Next contour
        i_cnt = hierarchy[0][i_cnt][0]

    return classes, Centers

if __name__ == '__main__':
    vid_name = 'input_video.avi'
    vid = cv2.VideoCapture(vid_name)
    ret, blackframe = vid.read()

    key = None

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('output_video.avi', fourcc, 20, (int(blackframe.shape[1] * 0.8), int(blackframe.shape[0] * 0.8)))

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

        # Classifier
        classes, Centers = classifyObjs(mask)
        classes_gray = cv2.cvtColor(classes, cv2.COLOR_BGR2GRAY)
        frm_classes = np.copy(classes)
        frm_classes[classes_gray == 0, :] = frame_bgr[classes_gray == 0, :]

        # Triangles descriptions
        text_w, text_h = 60, 16
        for (x, y) in Centers[0]:
            cv2.rectangle(frm_classes, (int(x - text_w / 2), int(y - text_h / 2)), (int(x + text_w / 2), int(y + text_h / 2)), (255, 255, 255), -1)
            cv2.putText(frm_classes, 'triangle', (int(x - text_w / 2), int(y + text_h / 2) - 3), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 0), 1)

        # Circle descriptions
        text_w, text_h = 44, 16
        for (x, y) in Centers[1]:
            cv2.rectangle(frm_classes, (int(x - text_w / 2), int(y - text_h / 2)),
                          (int(x + text_w / 2), int(y + text_h / 2)), (255, 255, 255), -1)
            cv2.putText(frm_classes, 'circle', (int(x - text_w / 2), int(y + text_h / 2) - 3), cv2.QT_FONT_NORMAL,
                        0.5, (0, 0, 0), 1)

        # Rectangle descriptions
        text_w, text_h = 74, 16
        for (x, y) in Centers[2]:
            cv2.rectangle(frm_classes, (int(x - text_w / 2), int(y - text_h / 2)),
                          (int(x + text_w / 2), int(y + text_h / 2)), (255, 255, 255), -1)
            cv2.putText(frm_classes, 'rectangle', (int(x - text_w / 2), int(y + text_h / 2) - 3), cv2.QT_FONT_NORMAL,
                        0.5, (0, 0, 0), 1)

        # Upd screen
        cv2.imshow('Objects classifier', frm_classes)

        if key == ord(' '):
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(25)

        # Write frame in output video
        frame_out = np.zeros((int(blackframe.shape[0] * 0.8), int(blackframe.shape[1] * 0.8), blackframe.shape[2]), dtype='uint8')
        frame_out[int(60 * 0.8):int(420 * 0.8)] = frm_classes
        writer.write(frame_out)

        # Get the next frame
        ret, blackframe = vid.read()
    vid.release()
    writer.release()
    cv2.destroyAllWindows()