import os
import cv2
import numpy as np
import sklearn


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
    train_portion = int(files_n * train_portion)
    val_portion = int(files_n * val_portion)

    # Randomly divide the dataset into 3 parts according to the given proportions
    filenames = sklearn.utils.shuffle(filenames, random_state=0)
    trainset = filenames[:train_portion]
    valset = filenames[train_portion:train_portion + val_portion]
    testset = filenames[train_portion + val_portion:]


if __name__ == '__main__':
    divide_dataset(0.7, 0.15)
