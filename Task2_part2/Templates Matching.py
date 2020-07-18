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


    # filenames = ['006', '007', '016']
    #
    # for i in range(3):
    #     # Read the template
    #     tmp_name = 'symbols/' + filenames[i] + '.png'
    #     tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)
    #
    #     # Template matching
    #     res = cv2.matchTemplate(img_gray, tmp, cv2.TM_CCOEFF_NORMED)
    #
    #     # Image to mark the found symbols on
    #     background = np.copy(img_bgr)
    #
    #     loc = np.where(res >= 0.8)
    #     for pt in zip(*loc[::-1]):
    #         # Draw the found symbols
    #         cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
    #         # Erase them on the pic
    #         cv2.rectangle(img_gray, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [255, 255, 255], -1)
    #

    # # 001
    # img_gray_bUp = np.copy(img_gray)
    # # Image to mark the found symbols on
    # background = np.copy(img_bgr)
    #
    # for i in range(4):
    #     methods = ['cv2.TM_CCOEFF', 'cv2.TM_SQDIFF',
    #                'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF']
    #     thresholds = [5200000, 35, 38, 35]  # left and right ain't perfect
    #
    #     if i == 0:
    #         # Read the template
    #         tmp_name = 'symbols/001.png'
    #         tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)
    #         tmp = cv2.rotate(tmp, cv2.ROTATE_180)
    #     else:
    #         tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)
    #
    #     # Template matching
    #     meth = eval(methods[i])
    #     if meth in [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]:
    #         res = cv2.matchTemplate(img_gray_bUp, tmp, meth, mask=getMask(tmp))
    #     else:
    #         res = cv2.matchTemplate(img_gray_bUp, tmp, meth)
    #
    #     if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #         loc = np.where(res <= thresholds[i])
    #     else:
    #         loc = np.where(res >= thresholds[i])
    #
    #     for pt in zip(*loc[::-1]):
    #         # Draw the found symbols
    #         cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
    #         # Erase them on the pic
    #         img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] = \
    #             cv2.bitwise_or(img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))
    #         if i == 0 or i == 2:
    #             img_gray_bUp[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] = \
    #                 cv2.bitwise_or(img_gray_bUp[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))

    # 004
    # Image to mark the found symbols on
    background = np.copy(img_bgr)
    for i in range(4):
        methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF']
        thresholds = [25, 25, 25, 25]

        if i == 0:
            # Read the template
            tmp_name = 'symbols/004.png'
            tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)
        else:
            tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)

        # Template matching
        meth = eval(methods[i])
        if meth in [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]:
            res = cv2.matchTemplate(img_gray, tmp, meth, mask=getMask(tmp))
        else:
            res = cv2.matchTemplate(img_gray, tmp, meth)

        if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = np.where(res <= thresholds[i])
        else:
            loc = np.where(res >= thresholds[i])

        for pt in zip(*loc[::-1]):
            # Draw the found symbols
            cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
            # Erase them on the pic
            img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] = \
                cv2.bitwise_or(img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))

    # 002
    img_gray_bUp = np.copy(img_gray)
    # Image to mark the found symbols on
    background = np.copy(img_bgr)
    for i in range(4):
        methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF']
        thresholds = [35, 35, 40, 20]

        if i == 0:
            # Read the template
            tmp_name = 'symbols/002.png'
            tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)
        else:
            tmp = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Template matching
        meth = eval(methods[i])
        if meth in [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]:
            res = cv2.matchTemplate(img_gray_bUp, tmp, meth, mask=getMask(tmp))
        else:
            res = cv2.matchTemplate(img_gray_bUp, tmp, meth)

        if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = np.where(res <= thresholds[i])
        else:
            loc = np.where(res >= thresholds[i])

        for pt in zip(*loc[::-1]):
            # Draw the found symbols
            cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
            # Erase them on the pic
            img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] = \
                cv2.bitwise_or(img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))

    # 003
    # Image to mark the found symbols on
    background = np.copy(img_bgr)
    for i in range(4):
        methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF']
        thresholds = [23, 23, 23, 23]

        if i == 0:
            # Read the template
            tmp_name = 'symbols/003.png'
            tmp = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE)
        else:
            tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)

        # Template matching
        meth = eval(methods[i])
        if meth in [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]:
            res = cv2.matchTemplate(img_gray, tmp, meth, mask=getMask(tmp))
        else:
            res = cv2.matchTemplate(img_gray, tmp, meth)

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
            img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] = \
                cv2.bitwise_or(img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))
        print(cnt)

    plt.imshow(background)
    plt.show()

    #008
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

        for pt in zip(*loc[::-1]):
            # Draw the found symbols
            cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
            # Erase them on the pic
            img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] =\
                cv2.bitwise_or(img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))

    #005
    for i in range(4):
        methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF']
        thresholds = [25, 25, 25, 25]

        if i == 0:
            # Read the template
            tmp_name = 'symbols/005.png'
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

        for pt in zip(*loc[::-1]):
            # Draw the found symbols
            cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
            # Erase them on the pic
            img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] =\
                    cv2.bitwise_or(img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))


    #009
    for i in range(4):
        methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        thresholds = [25, 25, 25, 0.195]

        if i == 0:
            # Read the template
            tmp_name = 'symbols/009.png'
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
            # Don't erase them on the pic because they could intersect with other symbols

    #015
    for i in range(4):
        methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF']
        thresholds = [50, 65, 68, 70]

        if i == 0:
            # Read the template
            tmp_name = 'symbols/015.png'
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

        for pt in zip(*loc[::-1]):
            # Draw the found symbols
            cv2.rectangle(background, pt, (pt[0] + tmp.shape[1], pt[1] + tmp.shape[0]), [0, 255, 0], 2)
            # Erase them on the pic
            img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]] =\
                    cv2.bitwise_or(img_gray[pt[1]:pt[1] + tmp.shape[0], pt[0]:pt[0] + tmp.shape[1]], getMask(tmp))



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