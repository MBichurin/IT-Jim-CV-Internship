import numpy as np
import csv
import glob
import random
import cv2
import torch
import torch.nn.functional as F
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# Initialise device to either CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Fixate randomization
random.seed(0)
torch.manual_seed(0)
# Datasets, dataloaders, model
trainset, valset, testset = None, None, None
train_loader, val_loader, test_loader = None, None, None
model = None
# Batch size, epochs number, classes number, channels number, images' height and width
batch_size = 32
n_epochs = 12
n_classes = 2
n_channels = 3
h, w = 46, 82


class MyDataset(Dataset):
    def __init__(self, filenames, transform):
        self.filenames = filenames
        self.transform = transform
        self.len = len(filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        # Read an image
        img = cv2.imread(self.filenames[item][0])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read a mask
        mask = cv2.imread(self.filenames[item][1])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255

        # Transforms
        transformed = self.transform(image=img, mask=mask)

        return transformed['image'], transformed['mask']


class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # (batch_size, n_channels, 46, 82)

        self.conv_L1 = torch.nn.Conv2d(n_channels, 32, kernel_size=3) # ==> (b_s, n_channels, 44, 80)
        # Max pooling ==> (b_s, 32, 22, 40)
        self.batchnorm_L1 = torch.nn.BatchNorm2d(32) # ==> same

        self.conv_L2 = torch.nn.Conv2d(32, 32, kernel_size=3) # ==> (b_s, 32, 20, 38)
        # Max pooling ==> (b_s, 32, 10, 19)
        self.batchnorm_L2 = torch.nn.BatchNorm2d(32) # ==> same

        self.upsample_L3 = torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2) # ==> (b_s, 16, 22, 40)
        # Add concatenation ==> (b_s, 48, 22, 40)
        self.batchnorm_L3 = torch.nn.BatchNorm2d(48) # ==> same

        self.upsample_L4 = torch.nn.ConvTranspose2d(48, 10, kernel_size=4, stride=2) # ==> (b_s, 10, 46, 82)
        self.batchnorm_L4 = torch.nn.BatchNorm2d(10) # ==> same

        self.conv_L5 = torch.nn.Conv2d(10, 1, kernel_size=1)  # ==> (b_s, 1, 46, 82)

        self.drop_L = torch.nn.Dropout(0.4)

    def forward(self, x):
        show_shapes = False

        # (batch_size, n_channels, 46, 82)
        if show_shapes:
            print(x.shape)

        x = F.leaky_relu(self.conv_L1(x)) # ==> (b_s, n_channels, 44, 80)
        if show_shapes:
            print(x.shape)
        x = F.max_pool2d(x, 2) # ==> (b_s, 32, 22, 40)
        if show_shapes:
            print(x.shape)
        x_1 = x.clone() # Save for skip connection
        x = self.batchnorm_L1(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.conv_L2(x)) # ==> (b_s, 32, 20, 38)
        if show_shapes:
            print(x.shape)
        x = F.max_pool2d(x, 2) # (b_s, 32, 10, 19)
        if show_shapes:
            print(x.shape)
        x = self.batchnorm_L2(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.upsample_L3(x)) # ==> (b_s, 16, 22, 40)
        if show_shapes:
            print(x.shape)
        x = torch.cat((x, x_1), 1) # ==> (b_s, 48, 22, 40)
        if show_shapes:
            print(x.shape)
        x = self.batchnorm_L3(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.upsample_L4(x)) # ==> (b_s, 10, 46, 82)
        if show_shapes:
            print(x.shape)
        x = self.batchnorm_L4(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.conv_L5(x)) # ==> (b_s, 1, 46, 82)
        if show_shapes:
            print(x.shape)

        # Get rid of channels dimension
        x = x.squeeze(dim=1)

        return x


def percent(float_num):
    return ("%.2f" % (float_num * 100)) + '%'


def show_metrics(true_masks, predictions):
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


def save_model(filename):
    torch.save(model.state_dict(), filename + '.pth')


def load_model(filename):
    global model
    create_model()
    model.load_state_dict(torch.load(filename + '.pth'))


def create_model():
    global model
    model = FCN().to(device)


def read_dataset():
    ''' Augmentations '''
    # Train augmentations
    transform_train = albu.Compose([
        albu.CoarseDropout(max_holes=100, max_height=5, max_width=5,
                           min_holes=10, min_height=1, min_width=1,
                           fill_value=255, p=0.9),
        albu.CoarseDropout(max_holes=100, max_height=5, max_width=5,
                           min_holes=10, min_height=1, min_width=1,
                           fill_value=0, p=0.9),
        albu.Resize(h, w),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        albu.Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    # Test augmentations
    transform_test = albu.Compose([
        albu.Resize(h, w),
        albu.Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    # Datasets sizes
    # sets_sizes = [11943, 3001, 2309]
    sets_sizes = [11943, 100, 2309]

    # Read trainset names
    with open('dataset\\train_set.csv', newline='\n') as file:
        train_names = list(csv.reader(file, delimiter=','))
    # Shuffle
    random.shuffle(train_names)
    # Reduce the size of trainset
    train_names = train_names[:sets_sizes[0]]

    # Read valset names
    with open('dataset\\val_set.csv', newline='\n') as file:
        val_names = list(csv.reader(file, delimiter=','))
    # Shuffle
    random.shuffle(val_names)
    # Reduce the size of valset
    val_names = val_names[:sets_sizes[1]]

    # Read testset names
    with open('dataset\\test_set.csv', newline='\n') as file:
        test_names = list(csv.reader(file, delimiter=','))
    # Shuffle
    random.shuffle(test_names)
    # Reduce the size of testset
    test_names = test_names[:sets_sizes[2]]

    # Initialize train-, val- and test-
    # -sets and -loaders
    global trainset, valset, testset, train_loader, val_loader, test_loader

    trainset = MyDataset(train_names, transform_train)
    valset = MyDataset(val_names, transform_test)
    testset = MyDataset(test_names, transform_test)

    train_loader = DataLoader(trainset, batch_size, shuffle=False, num_workers=1) # Already shuffled
    val_loader = DataLoader(valset, batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=1)

    print('Dataset names\'re read, loaders\'re created')


def tensor_to_bin_img(tensor):
    img = tensor.numpy()
    # print(np.amin(img), np.amax(img))
    img -= np.amin(img)
    # print(np.amin(img), np.amax(img))
    img = img / np.amax(img)
    # print(np.amin(img), np.amax(img))
    _, img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)
    # print(np.amin(img), np.amax(img))
    return img


def train():
    # Criterion, optimizer, epochs number
    criterion = torch.nn.BCEWithLogitsLoss().to(device) # Gotta find the best loss function for segmentation
    optimizer = torch.optim.Adam(model.parameters())

    # Iterate through epochs
    for epoch in range(n_epochs):
        print('Epoch ' + str(epoch + 1) + '/' + str(n_epochs) + ':')

        ''' Train '''
        print('  Training:')
        model.train(True)
        total_loss = 0
        batch_cnt = 0

        # Iterate through batches
        for i, (images, true_masks) in enumerate(train_loader):
            print('    batch #' + str(i + 1))
            images = images.to(device)
            true_masks = true_masks.to(device)

            # Get predictions
            predictions = model(images)

            # Loss function value
            loss = criterion(predictions, true_masks)

            # Update total_loss and batch_cnt
            total_loss += loss.item()
            batch_cnt += 1

            # Zero out parameter gradients
            optimizer.zero_grad()
            # Backward
            loss.backward()
            # Update weights
            optimizer.step()

        print('  Train loss = ' + str(total_loss / batch_cnt))


        ''' Validation '''
        print('  Validation:')
        total_loss = 0
        batch_cnt = 0
        model.train(False)

        # Iterate through batches
        for i, (images, true_masks) in enumerate(val_loader):
            print('    batch #' + str(i + 1))
            images = images.to(device)
            true_masks = true_masks.to(device)

            # Get predictions
            predictions = model(images)

            # Loss function value
            loss = criterion(predictions, true_masks)

            # Update total_loss and batch_cnt
            total_loss += loss.item()
            batch_cnt += 1

            # Zero out parameter gradients
            optimizer.zero_grad()

        print('  Validation loss = ' + str(total_loss / batch_cnt))


def test():
    # print('  Testing:')
    global model
    model.eval()

    test_predicts = np.zeros((len(testset), h, w), dtype=np.float)
    test_true_masks = np.zeros((len(testset), h, w), dtype=np.float)
    test_images = np.zeros((len(testset), h, w, 3), dtype=np.float)
    pic_ind = 0
    with torch.no_grad():
        # Iterate through batches
        for i, (images, true_masks) in enumerate(test_loader):
            # print('    batch #' + str(i + 1))
            images = images.to(device)
            true_masks = true_masks.to(device)

            # Get predictions
            predictions = model(images)

            # Remember ground truth and prediction of an image
            test_true_masks[pic_ind:pic_ind + true_masks.shape[0]] = true_masks

            # test_predicts[pic_ind:pic_ind + true_masks.shape[0]] = predictions.numpy()
            for i, prediction in enumerate(predictions, pic_ind):
                test_predicts[i] = tensor_to_bin_img(prediction)

            test_images[pic_ind:pic_ind + true_masks.shape[0]] = np.transpose(images.numpy(), (0, 2, 3, 1))

            # Update image index
            pic_ind += true_masks.shape[0]

    print('Testing\'s completed')

    for img, true_mask, pred in zip(test_images, test_true_masks, test_predicts):
        cv2.imshow('Image', img)
        cv2.imshow('Expected', true_mask)
        cv2.imshow('Result', pred)
        cv2.waitKey(0)

    # Show metrics
    show_metrics(test_true_masks, test_predicts)


def rewrite_csv():
    # The numbers of names in original CSV-files were bigger than actual number of files
    # + there were images and masks with same names but different file formats

    train_pics_names = glob.glob('dataset/train_set/*')
    train_masks_names = glob.glob('dataset/train_set_mask/*')
    with open('dataset\\train_set.csv', 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter=',')
        for pic, mask in zip(train_pics_names, train_masks_names):
            writer.writerow([pic, mask])

    val_pics_names = glob.glob('dataset/val_set/*')
    val_masks_names = glob.glob('dataset/val_set_mask/*')
    with open('dataset\\val_set.csv', 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter=',')
        for pic, mask in zip(val_pics_names, val_masks_names):
            writer.writerow([pic, mask])

    test_pics_names = glob.glob('dataset/test_set/*')
    test_masks_names = glob.glob('dataset/test_set_mask/*')
    with open('dataset\\test_set.csv', 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter=',')
        for pic, mask in zip(test_pics_names, test_masks_names):
            writer.writerow([pic, mask])


if __name__ == '__main__':
    src = 'load' # 'load' or 'create'

    # rewrite_csv()
    read_dataset()

    if src == 'create':
        print('Training a model:')
        create_model()
        train()
        save_model('fcn')
    else:
        print('The model\'s loaded')
        load_model('fcn')

    test()
