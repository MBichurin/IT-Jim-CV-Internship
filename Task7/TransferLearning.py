import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import cv2
from PIL import Image


# Datasets, dataloaders, model
trainset, valset, testset = None, None, None
train_loader, val_loader, test_loader = None, None, None
model = None
# Initialise device to either CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Number of classes, n of channels
n_classes = 10
n_channels = 1
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


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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

        self.drop_L = torch.nn.Dropout(0.7)

    def forward(self, x):
        check_layer_shapes = False

        # (batch_size, n_channels, 28, 28)
        if check_layer_shapes:
            print(x.shape)
        x = F.relu(self.conv_L1(x)) # ==> (b_s, 64, 28, 28)
        if check_layer_shapes:
            print(x.shape)
        x = F.max_pool2d(x, 2) # ==> (b_s, 64, 14, 14)
        if check_layer_shapes:
            print(x.shape)
        x = self.batchnorm_L1(x)
        x = self.drop_L(x)

        x = F.relu(self.conv_L2(x))  # ==> (b_s, 32, 14, 14)
        if check_layer_shapes:
            print(x.shape)
        x = self.batchnorm_L2(x)
        x = self.drop_L(x)

        x = F.relu(self.conv_L3(x))  # ==> (b_s, 16, 14, 14)
        if check_layer_shapes:
            print(x.shape)
        x = F.max_pool2d(x, 2) # ==> (b_s, 16, 7, 7)
        if check_layer_shapes:
            print(x.shape)
        x = self.batchnorm_L3(x)  # ==> same

        x = x.view(-1, 784) # ==> (b_s, 16 * 7 * 7) == (b_s, 784)
        if check_layer_shapes:
            print(x.shape)

        x = F.relu(self.dense_L1(x))  # ==> (b_s, 128)
        if check_layer_shapes:
            print(x.shape)
        x = self.drop_L(x)

        x = F.log_softmax(self.dense_L2(x))  # ==> (b_s, 10)
        if check_layer_shapes:
            print(x.shape)

        return x


def percent(float_num):
    return ("%.2f" % (float_num * 100)) + '%'


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
            predictions = np.argmax(predictions, 1)
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


def data_loader():
    batch_size = 32
    # Transformations for train- and testset
    # transform_train = albu.Compose([
    #     albu.Rotate(limit=(90, 90), always_apply=True),
    #     albu.CoarseDropout(max_holes=8, max_height=3, max_width=3,
    #                        min_holes=1, min_height=1, min_width=1,
    #                        fill_value=255, p=0.9),
    #     albu.CoarseDropout(max_holes=8, max_height=3, max_width=3,
    #                        min_holes=1, min_height=1, min_width=1,
    #                        fill_value=0, p=0.9),
    #     albu.Normalize(mean=0.5, std=0.5, always_apply=True),
    #     albu.pytorch.ToTensorV2(always_apply=True)
    # ])
    # transform_test = albu.Compose([
    #     albu.Normalize(mean=0.5, std=0.5, always_apply=True),
    #     albu.pytorch.ToTensorV2(always_apply=True)
    # ])

    transform_train = transforms.Compose(
        [transforms.RandomRotation((90, 90)),
        transforms.RandomPerspective(p=0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)] +
        [transforms.RandomErasing(p=0.7, scale=(0.002, 0.002), ratio=(1, 1), value=0)] * 10 +
        [transforms.RandomErasing(p=0.7, scale=(0.002, 0.002), ratio=(1, 1), value=1)] * 10
    )
    transform_test = transforms.Compose([
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Fixate randomization
    random.seed(0)
    torch.manual_seed(0)

    # Initialise dataset
    global trainset, valset, testset
    trainset = torchvision.datasets.mnist.MNIST(root='./data', train=True, download=True, transform=transform_train) # 60000

    trainset, valset, _ = torch.utils.data.random_split(trainset, [2000, 200, 57800]) # 58000, 2000
    testset = torchvision.datasets.mnist.MNIST(root='./data', train=False, download=True, transform=transform_test) # 10000
    testset, _ = torch.utils.data.random_split(testset, [200, 9800])

    # Create loaders
    global train_loader, val_loader, test_loader
    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, len(valset), shuffle=True, num_workers=1)
    test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=1)


def create_model():
    global model
    # Model, criterion, optimizer, epochs number
    model = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 30

    # Iterate through epochs
    for epoch in range(n_epochs):
        print('Epoch ' + str(epoch + 1) + '/' + str(n_epochs) + ':')

        ''' Train '''
        model.train(True)
        total_loss = 0
        batch_cnt = 0

        # Iterate through batches
        for data in train_loader:
            # Separate features and labels
            x_train, y_train = data
            # Wrap in torch Variables
            x_train, y_train = torch.autograd.Variable(x_train), torch.autograd.Variable(y_train)

            # Get predictions
            predictions = model(x_train)

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

        # Iterate through batches
        for data in val_loader:
            # Separate features and labels
            x_val, y_val = data
            # Wrap in torch Variables
            x_val, y_val = torch.autograd.Variable(x_val), torch.autograd.Variable(y_val)

            # Get predictions
            predictions = model(x_val)

            # Loss function value
            loss = criterion(predictions, y_val)
            # Update total_loss and batch_cnt
            total_loss += loss.item()
            batch_cnt += 1

            # Zero out parameter gradients
            optimizer.zero_grad()

        print('Validation loss = ' + str(total_loss / batch_cnt))


if __name__ == '__main__':
    data_loader()
    create_model()
