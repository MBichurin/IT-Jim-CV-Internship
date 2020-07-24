import numpy as np
import cv2


def orb_detector(img, tmp):
    # ORB
    orb = cv2.ORB_create(1000)
    # Keypoints and descriptors of img and tmp
    kp_img, des_img = orb.detectAndCompute(img, None)
    kp_tmp, des_tmp = orb.detectAndCompute(tmp, None)
    # Match descriptors of the pics
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = matcher.match(des_tmp, des_img)
    # Leave 10 best matches
    matches = sorted(matches, key=lambda x: x.distance)
    selected = []
    for m in matches:
        if m.distance > 30:
            break
        selected.append(m)

    if len(selected) > 10:
        # Reshape keypoints to pass them in cv2.findHomography
        img_pts = np.float32([kp_img[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
        tmp_pts = np.float32([kp_tmp[m.queryIdx].pt for m in selected]).reshape(-1, 1, 2)
        # Get the homography matrix to fit template on the frame and the matches mask
        hom_Mat, matchMask = cv2.findHomography(tmp_pts, img_pts, cv2.RANSAC, 5.0)
        matchMask = matchMask.ravel().tolist()
        # Get the corners of the tmp (not features but top-left, top-right etc.)
        h, w = tmp.shape[:2]
        frm_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        # Draw a rectangle on the detected chocolate
        rect = cv2.perspectiveTransform(frm_corners, hom_Mat)
        img = cv2.polylines(img, [np.int32(rect)], True, (0, 0, 255), 3, cv2.LINE_AA)

        draw_params = dict(matchColor=None,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchMask,  # draw only inliers
                           flags=2)
        inliers = cv2.drawMatches(tmp, kp_tmp, img, kp_img, selected, None, **draw_params)
        cv2.imshow('win', cv2.resize(inliers, (int(inliers.shape[1] * 0.5), int(inliers.shape[0] * 0.5))))
    else:
        cv2.imshow('win', np.zeros_like(img))


if __name__ == '__main__':
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

    while ret:
        frm_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        tmp_gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Win', cv2.resize(frm, (int(frm.shape[1] * 0.7), int(frm.shape[0] * 0.7))))

        orb_detector(frm, tmp)

        if key == ord(' '):
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(30)

        # Write frame in output video
        # writer.write(frm)

        # Get the next frame
        ret, frm = vid.read()
    vid.release()
    # writer.release()
    cv2.destroyAllWindows()