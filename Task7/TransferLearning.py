import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
HFCNames = ("Zero (0)", "One (1)", "Two (2)", "Three (3)", "Four (4)",
            "Five (5)", "Six (6)", "Seven (7)", "Eight (8)", "Nine (9)")
# Modes handles
ModeHandles = ('***************** “rotated” CNN on a rotated test dataset ******************',
               '****************** “rotated” CNN on a normal test dataset ******************',
               '**************** retrained CNN a) on a normal test dataset *****************',
               '**************** retrained CNN b) on a normal test dataset *****************',
               '**************** retrained CNN c) on a normal test dataset *****************')


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

        self.drop_L = torch.nn.Dropout(0.4)

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
        x = self.batchnorm_L3(x)

        x = x.view(-1, 784) # ==> (b_s, 16 * 7 * 7) == (b_s, 784)
        if check_layer_shapes:
            print(x.shape)

        x = F.relu(self.dense_L1(x))  # ==> (b_s, 128)
        if check_layer_shapes:
            print(x.shape)
        x = self.drop_L(x)

        x = F.log_softmax(self.dense_L2(x), dim=1)  # ==> (b_s, 10)
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


def save_model(filename):
    torch.save(model.state_dict(), filename + '.pth')


def load_model(filename):
    global model
    create_model()
    model.load_state_dict(torch.load(filename + '.pth'))


def data_loader(rotated):
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
        [transforms.RandomRotation((90, 90) if rotated else (0, 0)),
        transforms.RandomPerspective(p=0.9, distortion_scale=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)] +
        [transforms.RandomErasing(p=0.7, scale=(0.002, 0.002), ratio=(1, 1), value=0)] * 10 +
        [transforms.RandomErasing(p=0.7, scale=(0.002, 0.002), ratio=(1, 1), value=1)] * 10
    )
    transform_test = transforms.Compose([
        transforms.RandomRotation((90, 90) if rotated else (0, 0)),
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
    model = SimpleCNN().to(device)


def train():
    # Criterion, optimizer, epochs number
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

            # Get predictions
            predictions = model(x_train.to(device))

            # Loss function value
            loss = criterion(predictions, y_train.to(device))
            loss.requires_grad_(True)
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

            # Get predictions
            predictions = model(x_val.to(device))

            # Loss function value
            loss = criterion(predictions, y_val.to(device))
            # Update total_loss and batch_cnt
            total_loss += loss.item()
            batch_cnt += 1

            # Zero out parameter gradients
            optimizer.zero_grad()

        print('Validation loss = ' + str(total_loss / batch_cnt))


def test():
    global model
    model.eval()

    # Get predictions
    test_predicts = []
    with torch.no_grad():
        for data in test_loader:
            # Separate features and labels
            x_test, y_test = data
            # Predict
            predictions = model(x_test)
            predictions = np.argmax(predictions, 1)
            test_predicts.extend(predictions.numpy())

    # Get markers
    test_markers = np.zeros(len(testset), dtype=np.uint8)
    for i, (_, marker) in enumerate(testset):
        test_markers[i] = marker

    # Show metrics
    show_metrics(test_markers, test_predicts)


def print_handle(ind):
    if ind > 0:
        print('\n\n')
    print('****************************************************************************')
    print(ModeHandles[ind])
    print('****************************************************************************')


if __name__ == '__main__':
    src = ('load', 'load', 'load', 'load') # 'load' or 'create'

    ''' “rotated” CNN on a rotated test dataset '''
    print_handle(0)
    data_loader(rotated=True)
    if src[0] == 'create':
        create_model()
        train()
        save_model('rotated_cnn')
    else:
        load_model('rotated_cnn')
    test()

    ''' “rotated” CNN on a normal test dataset '''
    print_handle(1)
    data_loader(rotated=False)
    test()

    ''' retrained CNN a) on a normal test dataset '''
    print_handle(2)

    if src[1] == 'create':
        load_model('rotated_cnn')

        cnt = 0
        for child in model.children():
            cnt += 1
            if cnt == 8:
                break

            for param in child.parameters():
                param.requires_grad = False

        train()
        save_model('retrained_a')
    else:
        load_model('retrained_a')

    test()

    ''' retrained CNN b) on a normal test dataset '''
    print_handle(3)

    if src[2] == 'create':
        load_model('rotated_cnn')

        cnt = 0
        for child in model.children():
            cnt += 1
            if cnt == 7:
                break

            for param in child.parameters():
                param.requires_grad = False

        train()
        save_model('retrained_b')
    else:
        load_model('retrained_b')

    test()

    ''' retrained CNN c) on a normal test dataset '''
    print_handle(4)

    if src[3] == 'create':
        load_model('rotated_cnn')
        train()
        save_model('retrained_c')
    else:
        load_model('retrained_c')

    test()
