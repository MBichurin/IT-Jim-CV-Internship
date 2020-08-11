import os
import cv2
import numpy as np
import sklearn as sk
from sklearn import preprocessing, decomposition, ensemble, svm, gaussian_process, metrics


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
    max_prob = 0
    prob = np.zeros(n_classes)
    i = 0
    for (dist, file) in Dists:
        # Add 1 to the file's class
        train_class = dataset[file]
        prob[train_class] += 1

        # Update the test's class
        if prob[train_class] > max_prob:
            max_prob = prob[train_class]
            test_class = train_class

        i += 1
        if i == k:
            break
    prob /= k
    return test_class, prob


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


def validation_knn():
    global param
    max_prec = 0
    for cur_param in [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]:
        positives, probabilities = fit_knn(val_fts, valset, cur_param)
        prec = positives / val_fts.shape[0]
        print('For K=' + str(cur_param) + ' precision=' + percent(prec))
        if prec > max_prec:
            max_prec = prec
            param = cur_param


def fit_knn(fts, the_set, cur_param = None):
    positives = 0
    probabilities = np.zeros((fts.shape[0], n_classes))
    for i, (feature, file) in enumerate(zip(fts, the_set)):
        predict_class, prob = knn(feature, cur_param)
        if predict_class == dataset[file]:
            positives += 1
        probabilities[i] = prob
    return positives, probabilities


def precision(positives, total):
    return positives / total


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


def classifier(rand, fts, set, cl_type):
    # Define classifier
    if cl_type == 'rf':
        clf = ensemble.RandomForestClassifier(random_state=rand)
    if cl_type == 'svm':
        clf = svm.SVC(probability=True, random_state=rand)
    if cl_type == 'gauss':
        clf = gaussian_process.GaussianProcessClassifier(random_state=rand)
    # clf = tree.DecisionTreeClassifier(splitter='best', random_state=rand)

    # Fit the model and predict
    clf.fit(train_fts, train_classes)
    predictions = clf.predict(fts)
    probs = clf.predict_proba(fts)

    # Count positive predictions
    positives = 0
    for prediction, file in zip(predictions, set):
        if int(prediction) == dataset[file]:
            positives += 1
    return positives, probs, clf.classes_


def do_knn():
    print('\nKNN:')
    global param

    if validate[0]:
        # Validation
        validation_knn()
        print('Validation is done, the best K value is ' + str(param))
    else:
        param = 5

    # Testing
    positives, probs_knn = fit_knn(test_fts, testset, param)
    prec = precision(positives, test_fts.shape[0])
    print('  precision: ' + percent(prec))
    return probs_knn


def use_classifier(cl_type):
    if cl_type == 'rf':
        algo_idx = 1
    if cl_type == 'svm':
        algo_idx = 2
    if cl_type == 'gauss':
        algo_idx = 3

    if validate[algo_idx]:
        # Find the best random_state
        param = None
        max_prec = 0
        val_best_probs = None
        for rand in range(10):
            positives, val_probs, _ = classifier(rand, val_fts, valset, cl_type)
            prec = precision(positives, val_fts.shape[0])
            print('  For random_state=' + str(rand) + ' precision=' + percent(prec))
            if prec > max_prec:
                max_prec = prec
                param = rand
                val_best_probs = val_probs

        # Now we've chosen the best random_state
        print('  Validation is done, the best random_state value is ' + str(param))
    else:
        param = 0
        _, val_best_probs, _ = classifier(param, val_fts, valset, cl_type)

    # Fit the model and predict
    positives, probs_algo, classes_algo = classifier(param, test_fts, testset, cl_type)
    prec = precision(positives, test_fts.shape[0])
    print('  precision: ' + percent(prec))
    return probs_algo, val_best_probs, classes_algo


def normalize_and_dim_reduct(algo_idx):
    global train_fts, val_fts, test_fts

    # Normalize the features
    if normalize_method[algo_idx] == 1:
        train_fts, std_scale_hog, std_scale_loc_hist = normalize(train_fts)
        val_fts, std_scale_hog, std_scale_loc_hist = normalize(val_fts, std_scale_hog, std_scale_loc_hist)
        test_fts, std_scale_hog, std_scale_loc_hist = normalize(test_fts, std_scale_hog, std_scale_loc_hist)
    elif normalize_method[algo_idx] == 2:
        std_scale = preprocessing.StandardScaler().fit(train_fts)
        train_fts = std_scale.transform(train_fts)
        val_fts = std_scale.transform(val_fts)
        test_fts = std_scale.transform(test_fts)
    if normalize_method[algo_idx] != -1:
        print('  Features are normalized')

    if dim_red[algo_idx]:
        # Dimensionality reduction
        dim_reduction()


def visualise(predictions):
    for prediction, file in zip(predictions, testset):
        # Read an image
        img = cv2.imread(file)
        img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))
        # Background for the image and text
        main_img = 255 * np.ones((img.shape[1] + 80, img.shape[0] + 20, img.shape[2]), dtype=np.uint8)
        main_img[70:-10, 10:-10] = img

        cv2.putText(main_img, HFCNames[dataset[file]] + ' ==>',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(main_img, HFCNames[prediction],
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.imshow('Classifier', main_img)
        cv2.waitKey(0)


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
        metric_value[0] += (TP + TN) / (TP + FP + TN + FN) / n_classes
        # Error Rate
        metric_value[1] += (FP + FN) / (TP + FP + TN + FN) / n_classes
        # Macro-Precision
        metric_value[5] += TP / (TP + FP) / n_classes
        # Macro-Recall
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


if __name__ == '__main__':
    # Turn off validation for better testing
    # (default parameters may perform worse for other datasets)
    global validate
    validate = [False, False, False, False]
    # Normalize method, if -1 ==> features won't be normalized
    global normalize_method
    normalize_method = [-1, -1, 1]
    # Dimension reduction?
    global dim_red
    dim_red = [False, False, True]

    # Divide dataset into train-, validation- and testset
    divide_dataset(0.8, 0.1)

    # Get features of images in sets
    global train_fts, val_fts, test_fts
    train_fts = calc_fts(trainset)
    val_fts = calc_fts(valset)
    test_fts = calc_fts(testset)
    print('Features are calculated')


    # WASN'T HELPFUL ON THE ENSEMBLE VOTING STAGE
    # ''' KNN '''
    #
    # # Normalization and dimension reduction
    # normalize_and_dim_reduct(0)
    #
    # probs_knn = do_knn()


    ''' Random Forest '''

    print('Random Forest:')

    # Normalization and dimension reduction
    normalize_and_dim_reduct(1)

    # Get classes of the trainset
    global train_classes
    train_classes = np.zeros_like(trainset)
    for i, file in enumerate(trainset):
        train_classes[i] = dataset[file]
    probs_rf, val_probs_rf, classes_rf = use_classifier('rf')


    ''' Radial SVM '''

    print('Radial SVM:')

    # Normalization and dimension reduction
    normalize_and_dim_reduct(2)

    probs_svm, val_probs_svm, classes_svm = use_classifier('svm')


    # TOO SLOW FOR SUCH A BIG NUMBER OF FEATURES :_(
    # ''' Gaussian Process '''
    #
    # print('\nGaussian Process:')
    #
    # # Normalization and dimension reduction
    # normalize_and_dim_reduct(3)
    #
    # probs_gauss, classes_gauss = use_classifier('gauss')


    ''' Ensemble voting '''

    print('Ensemble voting:')
    # Validation
    if validate[3]:
        max_prec = 0
        param = None
        for param_rf in np.arange(0, 1.01, 0.05):
            param_svm = 1 - param_rf
            # Unite probabilities
            probs = np.zeros_like(probs_rf)
            for i in range(n_classes):
                probs[:, int(classes_rf[i])] += val_probs_rf[:, i] * param_rf
                probs[:, int(classes_svm[i])] += val_probs_svm[:, i] * param_svm

            predictions = np.argmax(probs, axis=1)

            # Count positive predictions
            positives = 0
            for prediction, file in zip(predictions, valset):
                if prediction == dataset[file]:
                    positives += 1
            prec = precision(positives, val_fts.shape[0])
            print('  For param_rf=' + str(param_rf) + ' precision=' + percent(prec))
            # Update param
            if prec > max_prec:
                max_prec = prec
                param = param_rf

        print('  Validation is done, the best param_rf value is ' + "%.2f" % param)
    else:
        param = 0.5

    # Testing
    param_rf = param
    param_svm = 1 - param_rf
    # Unite probabilities
    probs = np.zeros_like(probs_rf)
    for i in range(n_classes):
        probs[:, int(classes_rf[i])] += probs_rf[:, i] * param_rf
        probs[:, int(classes_svm[i])] += probs_svm[:, i] * param_svm

    predictions = np.argmax(probs, axis=1)

    # Count positive predictions
    positives = 0
    markers = np.zeros_like(predictions)
    for i, (prediction, file) in enumerate(zip(predictions, testset)):
        markers[i] = dataset[file]
        if prediction == markers[i]:
            positives += 1
    prec = precision(positives, test_fts.shape[0])
    print('  The solution\'s precision: ' + percent(prec))

    # Solution's metrics
    show_metrics(markers, predictions)
    # print(metrics.classification_report(markers, predictions, digits=4))


    # Show testing images and their predicted classes
    visualise(predictions)
