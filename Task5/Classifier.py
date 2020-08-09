import os
import cv2
import numpy as np
import sklearn as sk
from sklearn import preprocessing


def divide_dataset(train_portion, val_portion):
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
    train_size = int(files_n * train_portion)
    val_size = int(files_n * val_portion)

    # Randomly divide the dataset into 3 parts according to the given proportions
    filenames = sk.utils.shuffle(filenames, random_state=0)
    trainset = filenames[:train_size]
    valset = filenames[train_size:train_size + val_size]
    testset = filenames[train_size + val_size:]

    print('The dataset is divided: train - ' + percent(train_portion) + ', validation - ' + percent(val_portion) +
          ', test - ' + percent(1 - train_portion - val_portion))


def percent(float_num):
    return ("%.2f" % (float_num * 100)) + '%'


def calc_fts(set):
    # Features set
    fts_set = []
    # Number of features from hog
    global fts1_n
    # Iterate through filenames
    for filename in set:
        # Get HOG features of an image
        img = cv2.imread(filename)
        fts_hog = get_hog_fts(img)
        # Remember the number of HOG fts
        fts1_n = len(fts_hog)
        # Get Local Histograms features of an image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fts_loc_hist = get_segmhist_fts(img)
        # Add all the features to the features_set
        fts = np.concatenate((fts_hog, fts_loc_hist), axis=0)
        fts_set.extend([fts])
    return np.array(fts_set)


def normalize(fts_set, std_scale_hog=None, std_scale_loc_hist=None):
    # Reshape HOG and LocHist features into vectors
    hog_fts = fts_set[:, :fts1_n].reshape(-1, 1)
    loc_hist_fts = fts_set[:, fts1_n:].reshape(-1, 1)
    # If it's a trainset
    if std_scale_hog is None:
        std_scale_hog = preprocessing.StandardScaler().fit(hog_fts)
        std_scale_loc_hist = preprocessing.StandardScaler().fit(loc_hist_fts)
    # Standardize features
    hog_fts = std_scale_hog.transform(hog_fts)
    loc_hist_fts = std_scale_loc_hist.transform(loc_hist_fts)
    # Reshape features the way they were
    fts_set[:, :fts1_n] = hog_fts.reshape((fts_set.shape[0], fts1_n))
    fts_set[:, fts1_n:] = loc_hist_fts.reshape((fts_set.shape[0], fts_set.shape[1] - fts1_n))
    return fts_set, std_scale_hog, std_scale_loc_hist



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


def distance(A, B):
    chi_sq = np.sum(np.divide(np.power(np.subtract(A, B), 2), np.add(np.add(A, B), 1e-10))) / 2
    return chi_sq


def knn(test_feature, k):
    # (distance, filename)
    Dists = [(0, 0)] * train_fts.shape[0]
    # Iterate through the trainset
    for i, (train_feature, file) in enumerate(zip(train_fts, trainset)):
        Dists[i] = (distance(test_feature, train_feature), file)
    Dists = sorted(Dists)

    # Find the most popular class among K nearest neighbors
    test_class = None
    max_n = 0
    Classes = {}
    i = 0
    for (dist, file) in Dists:
        # Add 1 to the file's class
        train_class = dataset[file]
        if train_class in Classes:
            Classes[train_class] += 1
        else:
            Classes[train_class] = 1

        # Update the test's class
        if Classes[train_class] > max_n:
            max_n = Classes[train_class]
            test_class = train_class

        i += 1
        if i == k:
            break
    return test_class


# Human friendly class names
HFCNames = {
    "dataset\\n01855672": "Bird",
    "dataset\\n02091244": "Dog",
    "dataset\\n02114548": "Wolf",
    "dataset\\n02138441": "Meerkat",
    "dataset\\n02174001": "Bug",
    "dataset\\n02950826": "Cannon",
    "dataset\\n02971356": "Box",
    "dataset\\n02981792": "Ship",
    "dataset\\n03075370": "Lock",
    "dataset\\n03417042": "Garbage truck",
    "dataset\\n03535780": "Acrobat",
    "dataset\\n03584254": "mp3 player",
    "dataset\\n03770439": "Woman",
    "dataset\\n03773504": "Rocket",
    "dataset\\n03980874": "Strange scarf",
    "dataset\\n09256479": "Coral"
}


def validation():
    global param
    max_prec = 0
    for cur_param in [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]:
        prec = calc_precision(val_fts, valset, cur_param)
        print('For param=' + str(cur_param) + ' precision=' + percent(prec))
        if prec > max_prec:
            max_prec = prec
            param = cur_param


def calc_precision(fts, set, K):
    positives = 0
    for feature, file in zip(fts, set):
        predict_class = knn(feature, K)
        img = cv2.imread(file)
        if predict_class == dataset[file]:
            positives += 1
    return positives / fts.shape[0]


if __name__ == '__main__':
    # Divide dataset into train-, validation- and testset
    divide_dataset(0.7, 0.15)

    # Get features of images in sets
    global train_fts, val_fts, test_fts
    train_fts = calc_fts(trainset)
    val_fts = calc_fts(valset)
    test_fts = calc_fts(testset)
    print('Features are calculated')

    # Normalize the features

    # train_fts, std_scale_hog, std_scale_loc_hist = normalize(train_fts)
    # val_fts, std_scale_hog, std_scale_loc_hist = normalize(val_fts, std_scale_hog, std_scale_loc_hist)
    # test_fts, std_scale_hog, std_scale_loc_hist = normalize(test_fts, std_scale_hog, std_scale_loc_hist)

    # std_scale = preprocessing.StandardScaler().fit(train_fts)
    # train_fts = std_scale.transform(train_fts)
    # val_fts = std_scale.transform(val_fts)
    # test_fts = std_scale.transform(test_fts)

    # print('Features are normalized')

    # Validation
    validation()
    print('Validation is done, the best param is ' + str(param))

    # Testing
    prec = calc_precision(test_fts, testset, param)
    print('The solution\'s precision: ' + percent(prec))
