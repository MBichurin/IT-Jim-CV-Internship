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

        # Read a mask
        mask = cv2.imread(self.filenames[item][1])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255

        # Transforms
        transformed = self.transform(image=img, mask=mask)

        return transformed['image'], transformed['mask']


class InferDataset(Dataset):
    def __init__(self, filenames, transform):
        self.filenames = filenames
        self.transform = transform
        self.len = len(filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img = cv2.imread(self.filenames[item])
        return img, self.transform(image=img)['image']


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

        # # Convert to binary
        # for i, pic in enumerate(x):
        #     pic = pic - torch.min(pic)
        #     torch.true_divide(pic, torch.max(pic))
        #     x[i] = torch.where(pic > 0.5, torch.tensor(1.), torch.tensor(0.))

        return x


def percent(float_num):
    return ("%.2f" % (float_num * 100)) + '%'


def calc_intersection(gt_bbox, nn_bbox):
    rect1 = (gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3])
    rect2 = (nn_bbox[0], nn_bbox[1], nn_bbox[0] + nn_bbox[2], nn_bbox[1] + nn_bbox[3])

    rect_inter = (max(rect1[0], rect2[0]), max(rect1[1], rect2[1]),
                  min(rect1[2], rect2[2]), min(rect1[3], rect2[3]))
    return max((rect_inter[2] - rect_inter[0]) * (rect_inter[3] - rect_inter[1]), 0)


def show_metrics(true_bboxes, predictions):
    # True pos, false pos, true neg, false neg
    tp, fp, tn, fn = 0, 0, 0, 0
    for true_bbox, pred_bbox in zip(true_bboxes, predictions):
        # There are no GT and predicted bboxes
        if true_bbox[0] == -1 and pred_bbox[0] == -1:
            tn += 1
        # There is no GT bbox, but the NN found one
        elif true_bbox[0] == -1 and pred_bbox[0] != -1:
            fp += 1
        # There is GT bbox, but the NN didn't find it
        elif true_bbox[0] != -1 and pred_bbox[0] == -1:
            fn += 1
        else:
            # Calculate IoU
            intersection = calc_intersection(true_bbox, pred_bbox)
            union = true_bbox[2] * true_bbox[3] + pred_bbox[2] * pred_bbox[3] - intersection
            IoU = (intersection + 1e-6) / (union + 1e-6)

            if IoU >= 0.5:
                tp += 1
            else:
                fp += 1

    # Precision
    precision = tp / (tp + fp)
    print('Precision = ' + percent(precision))
    # Recall
    recall = tp / (tp + fn)
    print('Recall = ' + percent(recall))
    # F1
    F1 = 2 * precision * recall / (precision + recall)
    print('F1 = ' + percent(F1))


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

    img -= np.amin(img)
    img = img / np.amax(img)
    _, img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)

    return img


def train():
    # Criterion, optimizer, epochs number
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
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
            # loss = reversed_IOU(predictions, true_masks)
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


def get_bbox(mask):
    # Get outer contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # There are no or to many balls detected on the frame
    if len(contours) != 1:
        x, y, width, height = -1, -1, -1, -1
    # There is only 1 ball
    else:
        x, y, width, height = cv2.boundingRect(contours[0])

    return x, y, width, height


def test():
    # print('  Testing:')
    global model
    model.eval()

    test_predicts = np.zeros((len(testset), 4), dtype=np.int32)
    test_true_bboxes = np.zeros((len(testset), 4), dtype=np.int32)
    # test_images = np.zeros((len(testset), h, w, 3), dtype=np.float)
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
            # for i, (true_mask, prediction) in enumerate(zip(true_masks, predictions), pic_ind):
            for i, (true_mask, prediction) in enumerate(zip(true_masks, torch.sigmoid(predictions)), pic_ind):
                test_true_bboxes[i] = get_bbox(true_mask.numpy())
                test_predicts[i] = get_bbox(tensor_to_bin_img(prediction))

            # Update image index
            pic_ind += true_masks.shape[0]

    print('Testing\'s completed')

    # Show metrics
    show_metrics(test_true_bboxes, test_predicts)


def inference(path):
    print('Running inference...')

    # Test augmentations
    transform_infer = albu.Compose([
        albu.Resize(h, w),
        albu.Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    infer_names = glob.glob(path + '*')

    inferset = InferDataset(infer_names, transform_infer)

    infer_loader = DataLoader(inferset, batch_size=len(infer_names), shuffle=False, num_workers=1)

    global model
    model.eval()

    pic_ind = 0
    with torch.no_grad():
        # Iterate through batches
        for orig_images, images in infer_loader:
            images = images.to(device)
            orig_images = orig_images.numpy()

            # Get predictions
            predictions = model(images)

            # Remember images and predicted bboxes on them
            # for orig_img, prediction in zip(orig_images, predictions):
            for orig_img, prediction in zip(orig_images, torch.sigmoid(predictions)):
                x, y, width, height = get_bbox(tensor_to_bin_img(prediction))
                x = int(x * 640 / 82)
                width = int(width * 640 / 82)
                y = int(y * 360 / 46)
                height = int(height * 360 / 46)
                cv2.rectangle(orig_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.imshow('Ball detector', orig_img)
                cv2.waitKey(40)

            # Update image index
            pic_ind += images.shape[0]


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

    # test()

    inference('dataset/test_set/')
