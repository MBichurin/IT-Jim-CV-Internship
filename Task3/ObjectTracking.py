import numpy as np
import cv2

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
        cv2.imshow('Win', cv2.resize(frm, (int(frm.shape[1] * 0.7), int(frm.shape[0] * 0.7))))

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