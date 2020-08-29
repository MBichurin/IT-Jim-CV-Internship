import numpy as np
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
# Batch size
batch_size = 32


class MyDataset(Dataset):
    def __init__(self, pics, masks):
        self.pics = pics
        self.masks = masks
        self.len = len(pics)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.pics[item], self.pics[item]


class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        # (batch_size, n_channels, 28, 28)
        self.conv_L1 = torch.nn.Conv2d(n_channels, 64, kernel_size=3, padding=1) # ==> (b_s, 64, 28, 28)
        # Max pooling ==> (b_s, 64, 14, 14)
        self.batchnorm_L1 = torch.nn.BatchNorm2d(64) # ==> same

        self.conv_L2 = torch.nn.Conv2d(64, 32, kernel_size=3, padding=1) # ==> (b_s, 32, 14, 14)
        self.batchnorm_L2 = torch.nn.BatchNorm2d(32) # ==> same

        self.conv_L3 = torch.nn.Conv2d(32, 16, kernel_size=3, padding=1) # ==> (b_s, 16, 14, 14)
        # Max pooling ==> (b_s, 16, 7, 7)
        self.batchnorm_L3 = torch.nn.BatchNorm2d(16) # ==> same

        # Flatten ==> (b_s, 16 * 7 * 7) == (b_s, 784)
        self.dense_L1 = torch.nn.Linear(784, 128) # ==> (b_s, 128)
        self.dense_L2 = torch.nn.Linear(128, 10)  # ==> (b_s, 10)

        self.drop_L = torch.nn.Dropout(0.4)

    def forward(self, x):
        # (batch_size, n_channels, 28, 28)
        x = F.relu(self.conv_L1(x)) # ==> (b_s, 64, 28, 28)
        x = F.max_pool2d(x, 2) # ==> (b_s, 64, 14, 14)
        x = self.batchnorm_L1(x)
        x = self.drop_L(x)

        x = F.relu(self.conv_L2(x))  # ==> (b_s, 32, 14, 14)
        x = self.batchnorm_L2(x)
        x = self.drop_L(x)

        x = F.relu(self.conv_L3(x))  # ==> (b_s, 16, 14, 14)
        x = F.max_pool2d(x, 2) # ==> (b_s, 16, 7, 7)
        x = self.batchnorm_L3(x)

        x = x.view(-1, 784) # ==> (b_s, 16 * 7 * 7) == (b_s, 784)

        x = F.relu(self.dense_L1(x))  # ==> (b_s, 128)
        x = self.drop_L(x)

        x = F.log_softmax(self.dense_L2(x), dim=1)  # ==> (b_s, 10)

        return x


def read_pics_and_masks(set_name, transform):
    # Lists of names and of tensors
    pics_names = glob.glob('dataset/' + set_name + '_set/*')
    pics = []
    masks_names = glob.glob('dataset/' + set_name + '_set_mask/*')
    masks = []

    for i, (pic_name, mask_name) in enumerate(zip(pics_names, masks_names)):
        # Read an image
        img = cv2.imread(pic_name)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read a mask
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255

        transformed = transform(image=img, mask=mask)
        pics.extend(transformed['image'])
        masks.extend(transformed['mask'])

    return pics, masks


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
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        albu.Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    # Test augmentations
    transform_test = albu.Compose([
        albu.Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    # Read pictures and masks
    train_pics, train_masks = read_pics_and_masks('train', transform_train)
    val_pics, val_masks = read_pics_and_masks('val', transform_test)
    test_pics, test_masks = read_pics_and_masks('test', transform_test)

    # Initialize train-, val- and test-
    # -sets and -loaders
    global trainset, valset, testset, train_loader, val_loader, test_loader

    trainset = MyDataset(train_pics, train_masks)
    valset = MyDataset(val_pics, val_masks)
    testset = MyDataset(test_pics, test_masks)

    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=1)


if __name__ == '__main__':
    read_dataset()
