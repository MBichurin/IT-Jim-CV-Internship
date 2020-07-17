import cv2
import numpy as np

if __name__ == '__main__':
    img_name = 'plan.png'
    img_bgr = cv2.imread(img_name)

    cv2.imshow('Templates Matching', img_bgr)
    cv2.waitKey(0)