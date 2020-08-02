import numpy as np
import cv2
import os


def get_segmhist_fts(img):
    # Number of segments in a row/column; bins number; height and width
    segm_n = 7
    bins = (20, 2, 2)
    h, w = img.shape[:2]
    # Calculate segments
    segments = [(0, 0, 0, 0)] * (segm_n * segm_n)
    for j in range(segm_n):
        for i in range(segm_n):
            segments[j * segm_n + i] = (h * i // segm_n, w * j // segm_n, h * (i + 1) // segm_n, w * (j + 1) // segm_n)

    # Features vector
    features = []

    for (x0, y0, x1, y1) in segments:
        # Get the mask of the segment
        segm_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(segm_mask, (x0, y0), (x1, y1), 255, -1)

        # Calculate and normalize the segment's histogram
        hist = cv2.calcHist([img], [0, 1, 2], segm_mask, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features


# Filenames, features of the hole dataset
fts_set = []
filenames = []


def calc_dataset_fts():
    # Number of files
    global files_n
    files_n = 0
    # Iterate through all the files in the dataset
    for subdir, ris, files in os.walk('dataset'):
        for file in files:
            files_n += 1
            # Read an image and remember its name
            filename = os.path.join(subdir, file)
            filenames.extend([filename])
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Get features of the example image
            fts_set.extend([get_segmhist_fts(img)])
    print('The hole dataset\'s features are calculated!')


def distance(A, B, mode):
    if mode == 1:
        chi_sq = np.sum(np.divide(np.power(np.subtract(A, B), 2), np.add(np.add(A, B), 1e-10))) / 2
        return chi_sq
    if mode == 2:
        dist = np.sum(np.power(np.subtract(A, B), 2))
        return dist


def segm_hist_search(img_name):
    # Read the example image
    img_ex = cv2.imread(img_name)
    img_ex = cv2.cvtColor(img_ex, cv2.COLOR_BGR2HSV)
    # Features of the example image
    fts_ex = get_segmhist_fts(img_ex)

    # Calculate distances and sort them
    Dists = [(0, 0)] * files_n
    for i, (filename, fts) in enumerate(zip(filenames, fts_set)):
        Dists[i] = (distance(fts_ex, fts, 1), filename)
    print('The example image is compared with the dataset\'s images!')
    Dists = sorted(Dists)
    print('Sorted!')
    cv2.imshow('Ex', cv2.cvtColor(img_ex, cv2.COLOR_HSV2BGR))
    for i in range(5):
        print(Dists[i][0])
        match = cv2.imread(Dists[i][1])
        cv2.imshow('Top' + str(i + 1), match)
    cv2.waitKey(0)


# Number of words/clusters
words_n = 16 * 20
# ORB
orb = cv2.ORB_create(100)


def fill_vocabulary():
    # Descriptors
    descriptors = None

    i = 0

    file_n = 0

    # Iterate through all the files in the dataset
    for subdir, ris, files in os.walk('dataset'):
        for file in files:
            i += 1

            files_n += 1

            # Read an image and remember its name
            filename = os.path.join(subdir, file)
            filenames.extend([filename])
            img = cv2.imread(filename)

            '''Tried to describe corners using orb.compute ==> most of corners disappeared'''
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # corners = cv2.goodFeaturesToTrack(gray, 15, 0.2, 7)
            # if corners is not None:
            #     key = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in corners]
            #     print(key)
            #     key, des = orb.compute(gray, key)
            #     print(key)
            # else:
            #     key, des = orb.detectAndCompute(img, None)
            # key = orb.detect(gray)
            # print(key)
            # key, des = orb.compute(gray, key)
            # print(key)
            # img = cv2.drawKeypoints(img, key, img)

            key, des = orb.detectAndCompute(img, None)
            cv2.drawKeypoints(img, key, img)
            des = np.array(des)
            if des is not None:
                if descriptors is not None and des.shape != ():
                    descriptors = np.concatenate((descriptors, des))
                if descriptors is None:
                    descriptors = des

            # cv2.imshow('Win', cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4)))
            # cv2.waitKey(0)
        if i == 100:
            break

    # K-means
    descriptors = np.float32(descriptors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    global ret, label, center
    ret, label, center = cv2.kmeans(descriptors, words_n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


def get_cluster(des):
    cluster = None
    minDist = -1
    for i, c in enumerate(center):
        locDist = distance(des, c, 2)
        if minDist == -1 or locDist < minDist:
            minDist = locDist
            cluster = i
    return cluster


def describe_pic(img):
    word_percent = np.zeros(words_n)
    key, des = orb.detectAndCompute(img, None)
    des_n = 0
    for d in des:
        word = get_cluster(d)
        word_percent[word] += 1
        des_n += 1
    word_percent /= des_n
    return word_percent


def bow_search(img_name):
    # Read the example image and describe it using words
    img_ex = cv2.imread(img_name)
    word_percent_ex = describe_pic(img_ex)

    # Calculate distances and sort them
    Dists = [(0, 0)] * files_n
    for i, (filename, fts) in enumerate(zip(filenames, fts_set)):
        # Read an image and describe it using words
        img = cv2.imread(filename)
        word_percent = describe_pic(img)
        Dists[i] = (distance(fts_ex, fts, 1), filename)
    print('The example image is compared with the dataset\'s images!')
    Dists = sorted(Dists)
    print('Sorted!')
    cv2.imshow('Ex', cv2.cvtColor(img_ex, cv2.COLOR_HSV2BGR))
    for i in range(5):
        print(Dists[i][0])
        match = cv2.imread(Dists[i][1])
        cv2.imshow('Top' + str(i + 1), match)
    cv2.waitKey(0)

    i = 0

    # Iterate through all the files in the dataset
    for subdir, ris, files in os.walk('dataset'):
        for file in files:
            i += 1
            # Read an image and describe it using words
            filename = os.path.join(subdir, file)
            img = cv2.imread(filename)
            word_percent = describe_pic(img)


if __name__ == '__main__':
    ''' Local Histograms '''
    # calc_dataset_fts()
    # segm_hist_search('dataset/n02091244/n0209124400000005.jpg')

    ''' BoW '''
    fill_vocabulary()
    bow_search('dataset/n02091244/n0209124400000005.jpg')
