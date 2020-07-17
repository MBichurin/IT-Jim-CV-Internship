import cv2
import numpy as np
from matplotlib import pyplot as plt


def matcher():
    img_name = 'plan.png'
    img_bgr = cv2.imread(img_name)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fig = plt.figure(figsize=(6, 6))

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
    plt.show()


if __name__ == '__main__':
    matcher()
