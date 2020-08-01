import numpy as np
import cv2
import os


def get_features(img):
    # Number of segments in a row/column; bins number; height and width
    segm_n = 7
    bins = (8, 8, 3)
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
            fts_set.extend([get_features(img)])
    print('The hole dataset\'s features are calculated!')


def distance(A, B):
    chi_sq = np.sum([(a - b) * (a - b) / (a + b + 1e-10) for (a, b) in zip(A, B)]) / 2
    return chi_sq


def segm_hist_search(img_name):
    # Read the example image
    img_ex = cv2.imread(img_name)
    img_ex = cv2.cvtColor(img_ex, cv2.COLOR_BGR2HSV)
    # Features of the example image
    fts_ex = get_features(img_ex)

    # Calculate distances and sort them
    Dists = [(0, 0)] * files_n
    for i, (filename, fts) in enumerate(zip(filenames, fts_set)):
        Dists[i] = (distance(fts_ex, fts), filename)
    print('The example image is compared with the dataset\'s images!')
    Dists = sorted(Dists)
    print('Sorted!')
    cv2.imshow('Ex', cv2.cvtColor(img_ex, cv2.COLOR_HSV2BGR))
    for i in range(5):
        print(Dists[i][0])
        match = cv2.imread(Dists[i][1])
        cv2.imshow('Top' + str(i + 1), match)
    cv2.waitKey(0)


if __name__ == '__main__':
    calc_dataset_fts()
    segm_hist_search('dataset/n01855672/n0185567200000010.jpg')