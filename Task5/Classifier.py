import os
import cv2
import numpy as np
import sklearn as sk
from sklearn import preprocessing


def divide_dataset(train_size, val_size):
    # Names of dataset's files; their amount
    filenames = []
    files_n = 0
    # Dictionary <filename> -> <class> for the hole dataset;
    # Filenames of train set, validation set and test set
    global dataset, trainset, valset, testset
    dataset = {}

    # Iterate through all the files in the dataset
    for subdir, ris, files in os.walk('dataset'):
        for file in files:
            files_n += 1
            # Remember the img's name and class
            filename = os.path.join(subdir, file)
            filenames.extend([filename])
            dataset[filename] = subdir
    # Number of files in trainset and valset
    train_size = int(files_n * train_size)
    val_size = int(files_n * val_size)

    # Randomly divide the dataset into 3 parts according to the given proportions
    filenames = sk.utils.shuffle(filenames, random_state=0)
    trainset = filenames[:train_size]
    valset = filenames[train_size:train_size + val_size]
    testset = filenames[train_size + val_size:]


def calc_fts(set):
    # Features set
    fts_set = []
    # Iterate through filenames
    for filename in set:
        # Get HOG features of an image
        img = cv2.imread(filename)
        fts_hog = get_hog_fts(img)
        # Get Local Histograms features of an image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fts_loc_hist = get_segmhist_fts(img)
        # Add all the features to the features_set
        fts = np.concatenate((fts_hog, fts_loc_hist), axis=0)
        fts_set.extend([fts])
    return np.array(fts_set)


def get_hog_fts(img):
    # HOG
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9, 1, 4.0, 0, 0.2, 0,
                            cv2.HOGDESCRIPTOR_DEFAULT_NLEVELS)
    img = cv2.resize(img, (64, 64))
    des = hog.compute(img).reshape(-1)
    return des


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

    # Sliding window
    for (x0, y0, x1, y1) in segments:
        # Get the mask of the segment
        segm_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(segm_mask, (x0, y0), (x1, y1), 255, -1)

        # Calculate and normalize the segment's histogram
        hist = cv2.calcHist([img], [0, 1, 2], segm_mask, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features


if __name__ == '__main__':
    # Divide dataset into train-, validation- and testset
    divide_dataset(0.7, 0.15)

    # Get features of images in sets
    train_fts = calc_fts(trainset)
    val_fts = calc_fts(valset)
    test_fts = calc_fts(testset)

    # Normalize the features
    std_scale = preprocessing.StandardScaler().fit(train_fts)
    train_fts = std_scale.transform(train_fts)
    val_fts = std_scale.transform(val_fts)
    test_fts = std_scale.transform(test_fts)
