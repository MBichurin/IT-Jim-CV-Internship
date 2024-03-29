import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import tensorflow as tf
from sklearn import preprocessing, decomposition
from tensorflow.keras.models import Model as keras_Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input as keras_Input
from tensorflow.keras.layers import Dense as keras_Dense
from tensorflow.keras.layers import Conv2D as keras_Conv2D
from tensorflow.keras.layers import MaxPooling2D as keras_MaxPooling2D
from tensorflow.keras.layers import Flatten as keras_Flatten
from tensorflow.keras.optimizers import Adam as keras_Adam

import torch
from torch.utils.data import Dataset as torch_Dataset
from torch.utils.data import DataLoader as torch_DataLoader
import torch.nn.functional as F


n_classes = 16


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
            dataset[filename] = ClassIdx[subdir]
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


def get_markers(img_list):
    if package == 'keras':
        markers = np.zeros((len(img_list), n_classes), dtype=np.byte)
        for i, img in enumerate(img_list):
            markers[i][dataset[img]] = 1

    if package == 'torch':
        markers = np.zeros(len(img_list), dtype=np.longlong)
        for i, img in enumerate(img_list):
            markers[i] = dataset[img]

    return markers


def percent(float_num):
    return ("%.2f" % (float_num * 100)) + '%'


def read_pics():
    global train_pics, val_pics, test_pics

    if package == 'keras':
        train_pics = np.zeros((len(trainset), 84, 84, 3), dtype=np.float32)
        val_pics = np.zeros((len(valset), 84, 84, 3), dtype=np.float32)
        test_pics = np.zeros((len(testset), 84, 84, 3), dtype=np.float32)

        for i, file in enumerate(trainset):
            train_pics[i] = cv2.imread(file) / 255
        for i, file in enumerate(valset):
            val_pics[i] = cv2.imread(file) / 255
        for i, file in enumerate(testset):
            test_pics[i] = cv2.imread(file) / 255

    if package == 'torch':
        train_pics = np.zeros((len(trainset), 3, 84, 84), dtype=np.float32)
        val_pics = np.zeros((len(valset), 3, 84, 84), dtype=np.float32)
        test_pics = np.zeros((len(testset), 3, 84, 84), dtype=np.float32)

        for i, file in enumerate(trainset):
            img = cv2.imread(file) / 255
            train_pics[i][0, :, :] = img[:, :, 0]
            train_pics[i][1, :, :] = img[:, :, 1]
            train_pics[i][2, :, :] = img[:, :, 2]
        for i, file in enumerate(valset):
            img = cv2.imread(file) / 255
            val_pics[i][0, :, :] = img[:, :, 0]
            val_pics[i][1, :, :] = img[:, :, 1]
            val_pics[i][2, :, :] = img[:, :, 2]
        for i, file in enumerate(testset):
            img = cv2.imread(file) / 255
            test_pics[i][0, :, :] = img[:, :, 0]
            test_pics[i][1, :, :] = img[:, :, 1]
            test_pics[i][2, :, :] = img[:, :, 2]


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


def dim_reduction():
    global train_fts, val_fts, test_fts
    # Remember previous number of fts
    fts_prev_cnt = train_fts.shape[1]
    # PCA
    pca = decomposition.PCA(random_state=0)
    # Fit PCA and transform train features
    train_fts = pca.fit_transform(train_fts)
    # Transform validation features
    val_fts = pca.transform(val_fts)
    # Transform test features
    test_fts = pca.transform(test_fts)
    print('  Dimensionality\'s reducted (N of fts: %s -> %s)' % (fts_prev_cnt, train_fts.shape[1]))


def show_metrics(markers, predictions):
    # Confusion Matrix, true pos, false pos, true neg, false neg
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.uint32)
    tp = np.zeros(n_classes, dtype=np.uint32)
    fp = np.zeros(n_classes, dtype=np.uint32)
    tn = np.zeros(n_classes, dtype=np.uint32)
    fn = np.zeros(n_classes, dtype=np.uint32)
    for real, pred in zip(markers, predictions):
        conf_mat[pred, real] += 1
        if real == pred:
            tp[real] += 1
        else:
            fn[real] += 1
            fp[pred] += 1
        for i in range(n_classes):
            if i != real and i != pred:
                tn[i] += 1

    # Calculate metrics' values
    metric_name = ['Average Accuracy', 'Error Rate', 'micro-Precision', 'micro-Recall', 'micro-Fscore',
                   'Macro-Precision', 'Macro-Recall', 'Macro-Fscore']
    metric_value = np.zeros(len(metric_name), dtype=np.float32)
    for TP, FP, TN, FN in zip(tp, fp, tn, fn):
        # Average Accuracy
        if TP + FP + TN + FN > 0:
            metric_value[0] += (TP + TN) / (TP + FP + TN + FN) / n_classes
        # Error Rate
        if TP + FP + TN + FN > 0:
            metric_value[1] += (FP + FN) / (TP + FP + TN + FN) / n_classes
        # Macro-Precision
        if TP + FP > 0:
            metric_value[5] += TP / (TP + FP) / n_classes
        # Macro-Recall
        if TP + FN > 0:
            metric_value[6] += TP / (TP + FN) / n_classes
    # micro-Precision
    metric_value[2] = np.sum(tp) / (np.sum(tp) + np.sum(fp))
    # micro-Recall
    metric_value[3] = np.sum(tp) / (np.sum(tp) + np.sum(fn))
    # micro-Fscore
    metric_value[4] = 2 * metric_value[2] * metric_value[3] / (metric_value[2] + metric_value[3])
    # Macro-Fscore
    metric_value[7] = 2 * metric_value[5] * metric_value[6] / (metric_value[5] + metric_value[6])

    # Show the confusion matrix
    # Hat
    hor_line = '+' + 10 * '-' + '+' + n_classes * (7 * '-' + '+')
    print('\nConfusion Matrix:\n' + hor_line)
    print('|\\' + 9 * ' ' + '|' + n_classes * (7 * ' ' + '|'))
    print('| \\' + ' Actual |', end='')
    for real in range(n_classes):
        space_idx = HFCNames[real].find(' ')
        if space_idx == -1:
            l = len(HFCNames[real])
            print(((7 - l) // 2 + (7 - l) % 2) * ' ' + HFCNames[real] + ((7 - l) // 2) * ' ' + '|', end='')
        else:
            l = space_idx
            print(((7 - l) // 2 + (7 - l) % 2) * ' ' + HFCNames[real][:space_idx] + ((7 - l) // 2) * ' ' + '|', end='')
    print('\n|  \\______ |', end='')
    for real in range(n_classes):
        space_idx = HFCNames[real].find(' ')
        if space_idx == -1:
            print(7 * ' ' + '|', end='')
        else:
            l = len(HFCNames[real]) - space_idx - 1
            print(((7 - l) // 2 + (7 - l) % 2) * ' ' + HFCNames[real][space_idx + 1:] + ((7 - l) // 2) * ' ' + '|', end='')
    print('\n|Predicted\\|' + n_classes * (7 * ' ' + '|'))
    print(hor_line)

    # Main part
    for pred in range(n_classes):
        # Prediction column (1 line)
        print('|', end='')
        space_idx = HFCNames[pred].find(' ')
        if space_idx == -1:
            print(10 * ' ' + '|', end='')
        else:
            l = space_idx
            print(((10 - l) // 2 + (10 - l) % 2) * ' ' + HFCNames[pred][:space_idx] + ((10 - l) // 2) * ' ' + '|', end='')

        # Values (1 line)
        print(n_classes * (7 * ' ' + '|'))

        # Prediction column (2 line)
        print('|', end='')
        if space_idx == -1:
            l = len(HFCNames[pred])
            print(((10 - l) // 2 + (10 - l) % 2) * ' ' + HFCNames[pred] + ((10 - l) // 2) * ' ' + '|', end='')
        else:
            l = len(HFCNames[pred]) - space_idx - 1
            print(((10 - l) // 2 + (10 - l) % 2) * ' ' + HFCNames[pred][space_idx + 1:] + ((10 - l) // 2) * ' ' + '|', end='')

        # Values (2 line)
        for real in range(n_classes):
            print('  ', end='')
            number = conf_mat[pred, real]
            digits_before = False
            for i in [100, 10, 1]:
                digit = number // i
                if digit == 0 and ~digits_before and i != 1:
                    print(' ', end='')
                else:
                    print(str(digit), end='')
                    digits_before = True
                number %= i
            print('  |', end='')
        print('\n' + hor_line)

    # Show the metrics
    print()
    for name, value in zip(metric_name, metric_value):
        print(name + ' = ' + percent(value))


# Human friendly class names
HFCNames = ["Bird", "Dog", "Wolf", "Meerkat", "Bug", "Cannon", "Box", "Ship", "Lock", "Garbage truck", "Acrobat",
            "mp3 player", "Woman", "Rocket", "Strange scarf", "Coral"]


# Indexes of classes
ClassIdx = {
    "dataset\\n01855672": 0,
    "dataset\\n02091244": 1,
    "dataset\\n02114548": 2,
    "dataset\\n02138441": 3,
    "dataset\\n02174001": 4,
    "dataset\\n02950826": 5,
    "dataset\\n02971356": 6,
    "dataset\\n02981792": 7,
    "dataset\\n03075370": 8,
    "dataset\\n03417042": 9,
    "dataset\\n03535780": 10,
    "dataset\\n03584254": 11,
    "dataset\\n03770439": 12,
    "dataset\\n03773504": 13,
    "dataset\\n03980874": 14,
    "dataset\\n09256479": 15
}


def compile_model(model, nn_type):
    gd_algo = keras_Adam()
    if nn_type == 'fcnn':
        model.compile(loss='categorical_crossentropy', optimizer=gd_algo, metrics=['accuracy'])
    if nn_type == 'cnn':
        model.compile(loss='categorical_crossentropy', optimizer=gd_algo, metrics=['accuracy'])

    return model


def create_model(nn_type):
    if nn_type == 'fcnn':
        # Don't show the number of images to our NN
        input_shape = train_fts.shape[1:]
        # Define the structure of a Neural Network
        input = keras_Input(shape=input_shape)

        hidden_layer = keras_Dense(256, activation='relu')(input)
        hidden_layer = keras_Dense(256, activation='relu')(hidden_layer)
        hidden_layer = keras_Dense(64, activation='relu')(hidden_layer)
        hidden_layer = keras_Dense(32, activation='relu')(hidden_layer)

        classify_layer = keras_Dense(n_classes, activation='softmax')(hidden_layer)
        # Create a model
        model = keras_Model(inputs=input, outputs=classify_layer)

    if nn_type == 'cnn':
        # Don't show the number of images to our NN
        input_shape = train_pics.shape[1:]
        # Define the structure of a Neural Network
        input = keras_Input(shape=input_shape)

        hidden_layer = keras_Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input)
        hidden_layer = keras_MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(hidden_layer)
        hidden_layer = keras_Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(hidden_layer)
        hidden_layer = keras_MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(hidden_layer)
        hidden_layer = keras_Flatten()(hidden_layer)

        hidden_layer = keras_Dense(64, activation='relu')(hidden_layer)
        hidden_layer = keras_Dense(32, activation='relu')(hidden_layer)

        classify_layer = keras_Dense(n_classes, activation='softmax')(hidden_layer)

        # Create a model
        model = keras_Model(inputs=input, outputs=classify_layer)

    model = compile_model(model, nn_type)
    return model


def train(model, train_fts, train_markers, val_fts, val_markers, batch_size, epochs):
    history = model.fit(train_fts, train_markers, batch_size, epochs, validation_data=(val_fts, val_markers))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['train', 'valid'], loc='best')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'valid'], loc='best')
    plt.show()

    return model


def save_model(model, filename):
    if package == 'keras':
        file = open(filename + '.json', 'w')
        file.write(model.to_json())
        model.save_weights(filename + '.h5')

    if package == 'torch':
        torch.save(model.state_dict(), filename + '.pth')


def load_model(filename, nn_type):
    if package == 'keras':
        file = open(filename + '.json', 'r')
        model = model_from_json(file.read())
        model.load_weights(filename + '.h5')

    if package == 'torch':
        model = MyModel(nn_type)
        model.load_state_dict(torch.load(filename + '.pth'))

    return model


def test(model, fts, markers):
    predictions = model.predict(fts, verbose=0)
    predictions = tf.argmax(predictions, axis=1)
    markers = tf.argmax(markers, axis=1)
    show_metrics(markers, predictions)
    print()


def infer(model, nn_type, mode, name):
    print('Inference:')
    if mode == 'file':
        if nn_type == 'fcnn':
            fts = calc_fts([name])

        if nn_type == 'cnn':
            img = cv2.imread(name) / 255
            fts = np.expand_dims(img, axis=0)

        predictions = model.predict(fts, verbose=0)
        print('  "' + name + '":\n    ', end='')
        for i, class_prob in enumerate(predictions[0]):
            print(HFCNames[i] + ' - ' + percent(class_prob), end='; ')
        print()

    if mode == 'folder':
        # Iterate through all the files in the dataset
        for subdir, ris, files in os.walk(name):
            for file in files:
                filename = os.path.join(subdir, file)

                if nn_type == 'fcnn':
                    fts = calc_fts([filename])
                if nn_type == 'cnn':
                    img = cv2.imread(filename) / 255
                    fts = np.expand_dims(img, axis=0)

                predictions = model.predict(fts, verbose=0)
                print('  "' + filename + '":\n    ', end='')
                for i, class_prob in enumerate(predictions[0]):
                    print(HFCNames[i] + ' - ' + percent(class_prob), end='; ')
                print()


def infer_torch(model, mode, name):
    infer_dataset = InferDataset(nn_type, mode, name)
    infer_loader = torch_DataLoader(infer_dataset, 100, shuffle=False, num_workers=1)

    model.eval()

    all_probs = []

    with torch.no_grad():
        for data in infer_loader:
            # Separate features and labels
            x_infer, _ = data
            # Predict
            predictions = model(x_infer)
            all_probs.extend(predictions.numpy())


    print('Inference:')
    if mode == 'file':
        print('  "' + name + '":\n    ', end='')
        for i, class_prob in enumerate(all_probs[0]):
            print(HFCNames[i] + ' - ' + percent(class_prob), end='; ')
        print()

    if mode == 'folder':
        # File index
        file_ind = 0

        # Iterate through all the files in the dataset
        for subdir, ris, files in os.walk(name):
            for file in files:
                filename = os.path.join(subdir, file)

                print('  "' + filename + '":\n    ', end='')
                for i, class_prob in enumerate(all_probs[file_ind]):
                    print(HFCNames[i] + ' - ' + percent(class_prob), end='; ')
                print()

                file_ind += 1


class InferDataset(torch_Dataset):
    def __init__(self, nn_type, mode, name):
        if mode == 'file':
            if nn_type == 'fcnn':
                all_fts = calc_fts([name])

            if nn_type == 'cnn':
                all_fts = np.zeros((1, 3, 84, 84), dtype=np.float32)
                img = cv2.imread(name) / 255
                all_fts[0][0, :, :] = img[:, :, 0]
                all_fts[0][1, :, :] = img[:, :, 1]
                all_fts[0][2, :, :] = img[:, :, 2]

        if mode == 'folder':
            all_fts = []

            # Iterate through all the files in the dataset
            for subdir, ris, files in os.walk(name):
                for file in files:
                    filename = os.path.join(subdir, file)

                    if nn_type == 'fcnn':
                        fts = calc_fts([filename])
                    if nn_type == 'cnn':
                        fts = np.zeros((1, 3, 84, 84), dtype=np.float32)
                        img = cv2.imread(filename) / 255
                        all_fts[0][0, :, :] = img[:, :, 0]
                        all_fts[0][1, :, :] = img[:, :, 1]
                        all_fts[0][2, :, :] = img[:, :, 2]

                    all_fts.extend(fts)

        self.x_data = torch.FloatTensor(all_fts)

        self.len = len(self.x_data)

        self.y_data = np.zeros(self.len)

    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index])

    def __len__(self):
        return self.len


class MyDataset(torch_Dataset):
    def __init__(self, nn_type, set_type):
        if nn_type == 'fcnn':
            if set_type == 'train':
                self.x_data = torch.from_numpy(train_fts)
            if set_type == 'val':
                self.x_data = torch.from_numpy(val_fts)
            if set_type == 'test':
                self.x_data = torch.from_numpy(test_fts)

        if nn_type == 'cnn':
            if set_type == 'train':
                self.x_data = torch.from_numpy(train_pics)
            if set_type == 'val':
                self.x_data = torch.from_numpy(val_pics)
            if set_type == 'test':
                self.x_data = torch.from_numpy(test_pics)

        if set_type == 'train':
            self.y_data = torch.from_numpy(train_markers)
        if set_type == 'val':
            self.y_data = torch.from_numpy(val_markers)
        if set_type == 'test':
            self.y_data = torch.from_numpy(test_markers)

        self.len = len(self.x_data)

    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index])

    def __len__(self):
        return self.len


class MyModel(torch.nn.Module):
    def __init__(self, nn_type):
        self.nn_type = nn_type
        super(MyModel, self).__init__()
        if nn_type == 'fcnn':
            self.dense_L1 = torch.nn.Linear(train_fts.shape[1], 256)
            self.dense_L2 = torch.nn.Linear(256, 256)
            self.dense_L3 = torch.nn.Linear(256, 64)
            self.dense_L4 = torch.nn.Linear(64, 32)
            self.dense_L5 = torch.nn.Linear(32, n_classes)
        if nn_type == 'cnn':
            self.conv_L1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv_L2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.dense_L1 = torch.nn.Linear(32 * 21 * 21, 64)
            self.dense_L2 = torch.nn.Linear(64, 32)
            self.dense_L3 = torch.nn.Linear(32, n_classes)

    def forward(self, input):
        if self.nn_type == 'fcnn':
            hidden_layer = F.relu(self.dense_L1(input))
            hidden_layer = F.relu(self.dense_L2(hidden_layer))
            hidden_layer = F.relu(self.dense_L3(hidden_layer))
            hidden_layer = F.relu(self.dense_L4(hidden_layer))
            output = self.dense_L5(hidden_layer)

        if self.nn_type == 'cnn':
            hidden_layer = F.max_pool2d(F.relu(self.conv_L1(input)), 2)
            hidden_layer = F.max_pool2d(F.relu(self.conv_L2(hidden_layer)), 2)
            hidden_layer = hidden_layer.view(-1, 32 * 21 * 21)
            hidden_layer = F.relu(self.dense_L1(hidden_layer))
            hidden_layer = F.relu(self.dense_L2(hidden_layer))
            output = self.dense_L3(hidden_layer)

        output = F.softmax(output)
        # output = F.log_softmax(output)
        return output


def train_validate_torch(model, n_epochs, train_loader, val_loader):
    # Loss function, optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Iterate epochs
    for epoch in range(n_epochs):
        print('Epoch ' + str(epoch + 1) + '/' + str(n_epochs) + ':')

        ''' Train '''
        model.train(True)
        total_loss = 0
        batch_cnt = 0

        # Iterate batches
        for data in train_loader:
            # Separate features and labels
            x_train, y_train = data
            # Wrap in torch Variables; convert to float to calculate loss function
            x_train, y_train = torch.autograd.Variable(x_train), torch.autograd.Variable(y_train)

            # x_train, y_train = x_train.type(torch.FloatTensor), y_train.type(torch.LongTensor)

            # Get predictions
            predictions = model(x_train)

            # predictions = predictions.type(torch.FloatTensor)

            # Loss function value
            loss = criterion(predictions, y_train)
            # Update total_loss and batch_cnt
            total_loss += loss.item()
            batch_cnt += 1

            # Zero out parameter gradients
            optimizer.zero_grad()
            # Backward
            loss.backward()
            # Update weights
            optimizer.step()

        print('Train loss = ' + str(total_loss / batch_cnt))

        ''' Validation '''
        total_loss = 0
        batch_cnt = 0
        model.train(False)

        for data in val_loader:
            # Separate features and labels
            x_val, y_val = data
            # Wrap in torch Variables; convert to float to calculate loss function
            x_val, y_val = torch.autograd.Variable(x_val), torch.autograd.Variable(y_val)

            # x_val, y_val = x_train.type(torch.FloatTensor), y_train.type(torch.LongTensor)

            # Get predictions
            predictions = model(x_val)

            # predictions = predictions.type(torch.FloatTensor)

            # Loss function value
            loss = criterion(predictions, y_val)
            # Update total_loss and batch_cnt
            total_loss += loss.item()
            batch_cnt += 1

            # Zero out parameter gradients
            optimizer.zero_grad()

        print('Validation loss = ' + str(total_loss / batch_cnt))

    return model


def test_torch(model, test_loader):
    model.eval()

    all_predicts = []

    with torch.no_grad():
        for data in test_loader:
            # Separate features and labels
            x_test, y_test = data
            # Predict
            predictions = model(x_test)
            predictions = tf.argmax(predictions, 1)
            all_predicts.extend(predictions.numpy())

    show_metrics(test_markers, all_predicts)


def torch_main(nn_type, model_source):
    # Epochs numbers, batch size
    n_epochs = 30
    batch_size = 32

    # Data Loaders
    train_dataset = MyDataset(nn_type, 'train')
    train_loader = torch_DataLoader(train_dataset, batch_size, shuffle=False, num_workers=1)
    val_dataset = MyDataset(nn_type, 'val')
    val_loader = torch_DataLoader(val_dataset, batch_size, shuffle=False, num_workers=1)
    test_dataset = MyDataset(nn_type, 'test')
    test_loader = torch_DataLoader(test_dataset, batch_size, shuffle=False, num_workers=1)

    if model_source == 'create':
        # Create model
        model = MyModel(nn_type)

        # Train and validate
        model = train_validate_torch(model, n_epochs, train_loader, val_loader)

        # Save model
        save_model(model, nn_type)

    # Load model
    model = load_model(nn_type, nn_type)

    # Test
    test_torch(model, test_loader)

    # Infer
    # infer_torch(model, 'file', testset[0])
    infer_torch(model, 'folder', 'dataset\\n01855672')


# Package to use == 'keras' or 'torch'
package = 'keras'


if __name__ == '__main__':
    # Model type
    nn_type = 'fcnn'

    # Source of model == 'load' or 'create'
    model_source = 'load'

    # Divide dataset into train-, validation- and testset
    divide_dataset(0.8, 0.1)
    # Read and save images
    read_pics()
    # Get images' features sets
    global train_fts, val_fts, test_fts
    train_fts = calc_fts(trainset)
    val_fts = calc_fts(valset)
    test_fts = calc_fts(testset)
    print('Features are calculated')
    # Get one-hot markers of images
    global train_markers, val_markers, test_markers
    train_markers = get_markers(trainset)
    val_markers = get_markers(valset)
    test_markers = get_markers(testset)

    # Normalization and dimension reduction
    # train_fts, std_scale_hog, std_scale_loc_hist = normalize(train_fts)
    # val_fts, std_scale_hog, std_scale_loc_hist = normalize(val_fts, std_scale_hog, std_scale_loc_hist)
    # test_fts, std_scale_hog, std_scale_loc_hist = normalize(test_fts, std_scale_hog, std_scale_loc_hist)

    # std_scale = preprocessing.StandardScaler().fit(train_fts)
    # train_fts = std_scale.transform(train_fts)
    # val_fts = std_scale.transform(val_fts)
    # test_fts = std_scale.transform(test_fts)

    # dim_reduction()

    ''' If torch version is required '''
    if package == 'torch':
        torch_main(nn_type, model_source)
        quit(0)

    if model_source == 'create':
        # Create a model
        model = create_model(nn_type)

        # Train model
        if nn_type == 'fcnn':
            train(model, train_fts, train_markers, val_fts, val_markers, 32, 30)
        if nn_type == 'cnn':
            train(model, train_pics, train_markers, val_pics, val_markers, 128, 30)

        # Test model
        if nn_type == 'fcnn':
            test(model, test_fts, test_markers)
        if nn_type == 'cnn':
            test(model, test_pics, test_markers)

        # Infer
        infer(model, nn_type, 'file', testset[0])
        # infer(model, nn_type, 'folder', 'dataset\\n01855672')

        # Save model
        save_model(model, nn_type)

    # Load model
    model = load_model(nn_type, nn_type)
    model = compile_model(model, nn_type)

    # Test model
    if nn_type == 'fcnn':
        test(model, test_fts, test_markers)
    if nn_type == 'cnn':
        test(model, test_pics, test_markers)

    # Infer
    infer(model, nn_type, 'file', testset[0])
    # infer(model, nn_type, 'folder', 'dataset\\n01855672')