import cv2
import numpy as np
from matplotlib import pyplot as plt


def getMask(tmp):
    mask = cv2.bitwise_not(tmp)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(mask.shape, dtype=np.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], (255, 255, 255))

    return mask


def matcher():
    img_name = 'plan.png'
    img_bgr = cv2.imread(img_name)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    '''006'''
    # Read the template
    tmp_name = 'symbols/006.png'
    tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)

    # Template matching
    res = cv2.matchTemplate(img_gray, tmp, cv2.TM_CCOEFF_NORMED)

    # Image to mark the found symbols on
    background = np.copy(img_bgr)

    loc = np.where(res >= 0.7)
    for pt in zip(*loc[::-1]):
        # Draw the found symbols
        cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
        # Erase them on the pic
        cv2.rectangle(img_gray, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [255, 255, 255], -1)

    '''008'''
    for i in range(4):
        methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF']
        thresholds = [25, 25, 25, 25]

        if i == 0:
            # Read the template
            tmp_name = 'symbols/008.png'
            tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)
        else:
            tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)

        # Template matching
        meth = eval(methods[i])
        if meth in [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]:
            res = cv2.matchTemplate(img_gray, tmp, meth, mask=getMask(tmp))
        else:
            res = cv2.matchTemplate(img_gray, tmp, meth)

        # Image to mark the found symbols on
        background = np.copy(img_bgr)

        if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = np.where(res <= thresholds[i])
        else:
            loc = np.where(res >= thresholds[i])

        cnt = 0
        for pt in zip(*loc[::-1]):
            cnt += 1
            # Draw the found symbols
            cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
            # Erase them on the pic
            cv2.rectangle(img_gray, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [255, 255, 255], -1)
        print(cnt)


        plt.imshow(background)
        plt.show()

    '''for i in range(16):
        if i < 9:
            tmp_name = 'symbols/00' + str(i + 1) + '.png'
        else:
            tmp_name = 'symbols/0' + str(i + 1) + '.png'
        tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)

        res = cv2.matchTemplate(img_gray, tmp, cv2.TM_CCOEFF_NORMED)

        background = np.copy(img_bgr)

        loc = np.where(res >= 0.7)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)

        plt.imshow(background)
        plt.show()'''

    '''fig = plt.figure(figsize=(6, 6))

    for i in range(16):
        if i < 9:
            tmp_name = 'symbols/00' + str(i + 1) + '.png'
        else:
            tmp_name = 'symbols/0' + str(i + 1) + '.png'
        tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)

        # (tmp, 25, 0.01, 3)
        corners = cv2.goodFeaturesToTrack(tmp, 10, 0.01, 3)
        corners = np.int0(corners)

        tmp_bgr = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(tmp_bgr, (x, y), 3, (0, 0, 255), 1)

        axis = fig.add_subplot(4, 4, i + 1)
        axis.set_xticks([]), axis.set_yticks([])
        plt.imshow(tmp_bgr)
    plt.show()'''




if __name__ == '__main__':
    matcher()