import numpy as np
import cv2


def of_tracker(img, prev, prev_pts):
    # Optical Flow parameters
    lk_params = dict(winSize=(30, 30),
                     maxLevel=30,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # OF of previous frame's features on the next frame and backward
    pts, st, err = cv2.calcOpticalFlowPyrLK(prev, img, prev_pts, None, **lk_params)
    back_pts, st1, err1 = cv2.calcOpticalFlowPyrLK(img, prev, pts, None, **lk_params)

    # Reshape in list of pairs
    back_pts = back_pts.reshape(-1, 2)
    prev_pts = prev_pts.reshape(-1, 2)

    # Square of distances between prev frame's features and the (PREV)->(CURRENT)->(PREV) match
    dist_sq = (back_pts[:, 0] - prev_pts[:, 0]) ** 2 + (back_pts[:, 1] - prev_pts[:, 1]) ** 2
    dist_sq = dist_sq.reshape(-1, 1)

    # Use L1 or minEig errors to make a better match ==> didn't help
    # successful = ((st == 1) & (err < 40))

    # Use square of distances
    successful = ((st == 1) & (dist_sq < 1500))
    if np.sum(successful) == 0:
        return None, None

    return pts, successful


def orb_detector(img, tmp):
    # ORB
    orb = cv2.ORB_create(3000, 1.2, 30)
    # Keypoints and descriptors of img and tmp
    kp_img, des_img = orb.detectAndCompute(img, None)
    kp_tmp, des_tmp = orb.detectAndCompute(tmp, None)
    # Match descriptors of the pics (2 best matches)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    matches = matcher.knnMatch(des_tmp, des_img, k=2)
    # Lowe ratio test
    selected = []
    for m1, m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            selected.append(m1)

    if len(selected) >= 10:
        # Reshape matched keypoints to pass them in cv2.findHomography
        img_pts = np.float32([kp_img[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
        tmp_pts = np.float32([kp_tmp[m.queryIdx].pt for m in selected]).reshape(-1, 1, 2)

        return img_pts, tmp_pts
    else:
        return None, None


def boundingRect(img, prev, tmp, algo, prev_pts, prev_rect):
    # Just to play with features refreshing while Optical Flow works
    key = cv2.waitKey(1)

    # If the tracking is asked and there're features to track
    if (algo == 2) and (prev_pts is not None) and (key != ord('1')):
        # Get the new positions of features
        img_pts, successful = of_tracker(img, prev, prev_pts)

        prev_pts = prev_pts.reshape(-1, 1, 2)
        img_pts = img_pts.reshape(-1, 1, 2)

        if (img_pts is not None) and (len(img_pts) >= 10):
            # Get the homography matrix and the matches mask (using successful features only)
            hom_Mat, matchMask = cv2.findHomography(prev_pts[successful], img_pts[successful], cv2.RANSAC, 30.0)
            # Deform the rectangle on the detected chocolate
            rect = cv2.perspectiveTransform(prev_rect, hom_Mat)
            # Use the homography to keep features that weren't found on the new frame
            img_pts = cv2.perspectiveTransform(prev_pts, hom_Mat)
            img_pts = img_pts.reshape(-1, 2)

            # Draw the choco-rectangle
            if rect is not None:
                cv2.polylines(frm, [np.int32(rect)], True, (0, 255, 0), 3)
            # Draw the features
            # for x, y in img_pts:
            #     cv2.circle(frm, (x, y), 1, (255, 0, 0), 2)

            return img_pts, rect

    # Get ORB features
    img_pts, tmp_pts = orb_detector(img, tmp)
    if (img_pts is not None) and (tmp_pts is not None):
        # Get the homography matrix to fit template on the frame and the matches mask
        hom_Mat, matchMask = cv2.findHomography(tmp_pts, img_pts, cv2.RANSAC, 5.0)
        # Get the corners of the tmp (not features but top-left, top-right etc.)
        h, w = tmp.shape[:2]
        tmp_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        # Draw a rectangle on the detected chocolate
        rect = cv2.perspectiveTransform(tmp_corners, hom_Mat)
    else:
        rect = None

    # Draw the choco-rectangle
    if rect is not None:
        cv2.polylines(frm, [np.int32(rect)], True, (0, 0, 255), 3)

    return img_pts, rect


if __name__ == '__main__':
    # Choose the algorithm
    ALGO = 1

    # Load video
    vid_name = 'find_chocolate.mp4'
    vid = cv2.VideoCapture(vid_name)
    ret, frm = vid.read()
    # Load template/marker
    tmp_name = 'marker.jpg'
    tmp = cv2.imread(tmp_name)

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('output' + str(ALGO) + '.avi', fourcc, 20, (frm.shape[1], frm.shape[0]))

    # To track pressed buttons
    key = None

    # Prev frame
    prev = None
    # Prev frame's features and rectangle's coordinates
    prev_pts = None
    prev_rect = None

    while ret:
        # Current frame's backup
        frm_bUp = np.copy(frm)

        # Draw a bounding rectangle on the found chocolate
        prev_pts, prev_rect = boundingRect(frm, prev, tmp, ALGO, prev_pts, prev_rect)

        # Update previous frame
        prev = frm_bUp

        # Show and save
        cv2.imshow('Win', cv2.resize(frm, (int(frm.shape[1] * 0.7), int(frm.shape[0] * 0.7))))
        writer.write(frm)

        if key == ord(' '):
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)

        # Get the next frame
        ret, frm = vid.read()
    vid.release()
    writer.release()
    cv2.destroyAllWindows()