import numpy as np
import cv2


def of_tracker(img, prev, prev_pts):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=30,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # calculate optical flow
    pts, st, err = cv2.calcOpticalFlowPyrLK(prev, img, prev_pts, None, **lk_params)
    print(max(err * st))
    # successful = ((st == 1) & (err < 40))
    successful = (st == 1)
    if np.sum(successful) == 0:
        return None, None

    prev_pts = prev_pts.reshape(-1, 1, 2)
    pts = pts.reshape(-1, 1, 2)
    prev_pts = prev_pts[successful]
    pts = pts[successful]
    return prev_pts, pts


def orb_detector(img, tmp):
    # ORB
    orb = cv2.ORB_create(10000, 1.2, 40)
    # Keypoints and descriptors of img and tmp
    kp_img, des_img = orb.detectAndCompute(img, None)
    kp_tmp, des_tmp = orb.detectAndCompute(tmp, None)
    # Match descriptors of the pics
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = matcher.match(des_tmp, des_img)
    # Leave the best matches
    matches = sorted(matches, key=lambda x: x.distance)
    selected = []
    for m in matches:
        if m.distance > 30:
            break
        selected.append(m)

    if len(selected) >= 10:
        # Reshape matched keypoints to pass them in cv2.findHomography
        img_pts = np.float32([kp_img[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
        tmp_pts = np.float32([kp_tmp[m.queryIdx].pt for m in selected]).reshape(-1, 1, 2)

        return img_pts, tmp_pts
    else:
        return None, None


def boundingRect(img, prev, tmp, algo, prev_pts, prev_rect):
    key = cv2.waitKey(1)

    if (algo == 2) and (prev_pts is not None) and (key != ord('1')):
        prev_pts, img_pts = of_tracker(img, prev, prev_pts)

        if (img_pts is not None) and (len(img_pts) >= 10):
            # Get the homography matrix and the matches mask
            hom_Mat, matchMask = cv2.findHomography(prev_pts, img_pts, cv2.RANSAC, 10.0)
            # Deform the rectangle on the detected chocolate
            rect = cv2.perspectiveTransform(prev_rect, hom_Mat)

            if rect is not None:
                cv2.polylines(frm, [np.int32(rect)], True, (0, 255, 0), 3)

            return img_pts, rect

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

    if rect is not None:
        cv2.polylines(frm, [np.int32(rect)], True, (0, 0, 255), 3)

    return img_pts, rect




if __name__ == '__main__':
    # Choose the algorithm
    ALGO = 2

    # Load video
    vid_name = 'find_chocolate.mp4'
    vid = cv2.VideoCapture(vid_name)
    ret, frm = vid.read()

    # Load template/marker
    tmp_name = 'marker.jpg'
    tmp = cv2.imread(tmp_name)

    # Writer
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # writer = cv2.VideoWriter('output.mp4', fourcc, 20, (frm.shape[1], frm.shape[0]))

    key = None

    prev_pts = None
    prev_rect = None
    prev = None

    while ret:
        frm_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        tmp_gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        prev_pts, prev_rect = boundingRect(frm, prev, tmp, ALGO, prev_pts, prev_rect)

        prev = np.copy(frm)

        cv2.imshow('Win', cv2.resize(frm, (int(frm.shape[1] * 0.7), int(frm.shape[0] * 0.7))))

        if key == ord(' '):
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)

        # Write frame in output video
        # writer.write(frm)

        # Get the next frame
        ret, frm = vid.read()
    vid.release()
    # writer.release()
    cv2.destroyAllWindows()