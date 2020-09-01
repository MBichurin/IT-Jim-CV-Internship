import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Initialise device to either CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Fixate randomization
random.seed(0)
torch.manual_seed(0)
# Datasets, dataloaders, model
trainset, valset, inferset = [], [], []
train_loader, val_loader, infer_loader = None, None, None
model = None
# Batch size, epochs number, classes number, channels number, images' height and width
batch_size = 32
n_epochs = 12
n_classes = 10
n_channels = 1


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # (batch_size, n_channels, 28, 28)

        self.conv_L1 = torch.nn.Conv2d(n_channels, 32, kernel_size=3, padding=1) # ==> (b_s, 64, 28, 28)
        # Max pooling ==> (b_s, 32, 14, 14)
        # Save for skip connection
        self.batchnorm_L1 = torch.nn.BatchNorm2d(32) # ==> same

        self.conv_L2 = torch.nn.Conv2d(32, 16, kernel_size=3, padding=1) # ==> (b_s, 16, 14, 14)
        # Max pooling ==> (b_s, 16, 7, 7)
        # Save for skip connection
        self.batchnorm_L2 = torch.nn.BatchNorm2d(16) # ==> same

        self.conv_L3 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)  # ==> (b_s, 8, 7, 7)
        # Max pooling ==> (b_s, 8, 3, 3)
        # Save for skip connection
        self.batchnorm_L3 = torch.nn.BatchNorm2d(8)  # ==> same

        self.conv_L4 = torch.nn.Conv2d(8, 4, kernel_size=3, padding=1)  # ==> (b_s, 4, 3, 3)
        # Max pooling ==> (b_s, 4, 1, 1)
        self.batchnorm_L4 = torch.nn.BatchNorm2d(4)  # ==> same

        self.upsample_L5 = torch.nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2) # ==> (b_s, 8, 3, 3)
        # Add concatenation ==> (b_s, 16, 3, 3)
        self.batchnorm_L5 = torch.nn.BatchNorm2d(16) # ==> same

        self.upsample_L6 = torch.nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2) # ==> (b_s, 8, 7, 7)
        # Add concatenation ==> (b_s, 24, 7, 7)
        self.batchnorm_L6 = torch.nn.BatchNorm2d(24) # ==> same

        self.upsample_L7 = torch.nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2)  # ==> (b_s, 12, 14, 14)
        # Add concatenation ==> (b_s, 44, 14, 14)
        self.batchnorm_L7 = torch.nn.BatchNorm2d(44)  # ==> same

        self.upsample_L8 = torch.nn.ConvTranspose2d(44, 20, kernel_size=2, stride=2)  # ==> (b_s, 20, 28, 28)
        self.batchnorm_L8 = torch.nn.BatchNorm2d(20)  # ==> same

        self.conv_L9 = torch.nn.Conv2d(20, 6, kernel_size=1)  # ==> (b_s, 6, 28, 28)
        self.batchnorm_L9 = torch.nn.BatchNorm2d(6)  # ==> same

        self.conv_L10 = torch.nn.Conv2d(6, n_channels, kernel_size=1)  # ==> (b_s, n_channels, 28, 28)

        self.drop_L = torch.nn.Dropout(0.4)

    def forward(self, x):
        # (batch_size, n_channels, 28, 28)

        x = F.leaky_relu(self.conv_L1(x)) # ==> (b_s, 64, 28, 28)
        x = F.max_pool2d(x, 2) # ==> (b_s, 32, 14, 14)
        x_1 = x.clone() # Save for skip connection
        x = self.batchnorm_L1(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.conv_L2(x)) # ==> (b_s, 16, 14, 14)
        x = F.max_pool2d(x, 2) # ==> (b_s, 16, 7, 7)
        x_2 = x.clone() # Save for skip connection
        x = self.batchnorm_L2(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.conv_L3(x)) # ==> (b_s, 8, 7, 7)
        x = F.max_pool2d(x, 2) # M==> (b_s, 8, 3, 3)
        x_3 = x.clone() # Save for skip connection
        x = self.batchnorm_L3(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.conv_L4(x)) # ==> (b_s, 4, 3, 3)
        x = F.max_pool2d(x, 2) # ==> (b_s, 4, 1, 1)
        x = self.batchnorm_L4(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.upsample_L5(x)) # ==> (b_s, 8, 3, 3)
        x = torch.cat((x, x_3), 1) # ==> (b_s, 16, 3, 3)
        x = self.batchnorm_L5(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.upsample_L6(x)) # ==> (b_s, 8, 7, 7)
        x = torch.cat((x, x_2), 1) # ==> (b_s, 24, 7, 7)
        x = self.batchnorm_L6(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.upsample_L7(x)) # ==> (b_s, 12, 14, 14)
        x = torch.cat((x, x_1), 1) # ==> (b_s, 44, 14, 14)
        x = self.batchnorm_L7(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.upsample_L8(x)) # ==> (b_s, 20, 28, 28)
        x = self.batchnorm_L8(x)
        x = self.drop_L(x)

        x = F.leaky_relu(self.conv_L9(x)) # ==> (b_s, 6, 28, 28)
        x = self.batchnorm_L9(x)
        x = self.drop_L(x)

        x = self.conv_L10(x) # ==> (b_s, n_channels, 28, 28)

        return x


def percent(float_num):
    return ("%.2f" % (float_num * 100)) + '%'


def save_model(filename):
    torch.save(model.state_dict(), filename + '.pth')


def load_model(filename):
    global model
    create_model()
    model.load_state_dict(torch.load(filename + '.pth'))


def create_model():
    global model
    model = AutoEncoder().to(device)


def show_transform(data, transform_x, transform_y):
    for pil, _ in data:
        # Transform
        img_x = transform_x(pil)
        img_x = img_x.numpy().squeeze(0)
        img_y = transform_y(pil)
        img_y = img_y.numpy().squeeze(0)
        # Resize
        img_x = cv2.resize(img_x, (150, 150))
        img_y = cv2.resize(img_y, (150, 150))
        # Show
        cv2.imshow('Noised', img_x)
        cv2.imshow('Denoised', img_y)
        cv2.waitKey(0)


def read_dataset():
    ''' Augmentations '''
    # Train augmentations
    transform_x = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)] +
        [transforms.RandomErasing(p=1, scale=(0.0005, 0.0005), ratio=(1, 1), value=1)] * 100 +
        [transforms.RandomErasing(p=1, scale=(0.0005, 0.0005), ratio=(1, 1), value=0)] * 100
    )
    global transform_y
    transform_y = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)]
    )

    print('Reading data...')

    # Load MNIST images
    traindata = torchvision.datasets.mnist.MNIST(root='./data', train=True, download=True)
    traindata, _ = torch.utils.data.random_split(traindata, [10000, 50000])
    valdata = torchvision.datasets.mnist.MNIST(root='./data', train=False, download=True)
    valdata, inferdata, _ = torch.utils.data.random_split(valdata, [100, 1000, 8900])

    # show_transform(traindata, transform_x, transform_y)

    # Initialise dataset
    global trainset, valset, inferset
    if src == 'create':
        for img, _ in traindata:
            trainset.extend([[transform_x(img), transform_y(img)]])
        for img, _ in valdata:
            valset.extend([[transform_x(img), transform_y(img)]])
    for img, _ in inferdata:
        inferset.extend([[transform_x(img), transform_y(img)]])

    # Create loaders
    global train_loader, val_loader, infer_loader
    if src == 'create':
        train_loader = DataLoader(trainset, batch_size, shuffle=False, num_workers=1)
        val_loader = DataLoader(valset, len(valset), shuffle=False, num_workers=1)
    infer_loader = DataLoader(inferset, batch_size, shuffle=False, num_workers=1)

    print('Dataset\'s read')


def train():
    # Criterion, optimizer, epochs number
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Iterate through epochs
    for epoch in range(n_epochs):
        print('Epoch ' + str(epoch + 1) + '/' + str(n_epochs) + ':')

        ''' Train '''
        # print('  Training:')
        model.train(True)
        total_loss = 0
        batch_cnt = 0

        # Iterate through batches
        for i, (X, Y) in enumerate(train_loader):
            # print('    batch #' + str(i + 1))
            X = X.to(device)
            Y = Y.to(device)

            # Get predictions
            predictions = model(X)

            # Loss function value
            loss = criterion(predictions, Y)

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
        # print('  Validation:')
        total_loss = 0
        batch_cnt = 0
        model.train(False)

        # Iterate through batches
        for i, (X, Y) in enumerate(val_loader):
            # print('    batch #' + str(i + 1))
            X = X.to(device)
            Y = Y.to(device)

            # Get predictions
            predictions = model(X)

            # Loss function value
            loss = criterion(predictions, Y)

            # Update total_loss and batch_cnt
            total_loss += loss.item()
            batch_cnt += 1

            # Zero out parameter gradients
            optimizer.zero_grad()

        print('  Validation loss = ' + str(total_loss / batch_cnt))


def visualize_denoise(img_noise, img_pred):
    # Resize
    size = 150
    img_noise = cv2.resize(img_noise, (size, size))
    img_pred = cv2.resize(img_pred, (size, size))
    # Main image
    img = np.zeros((40 + size + 15, 3 * 15 + 2 * size), dtype=np.float)
    img[40:40 + size, 15:15 + size] = img_noise
    img[40:40 + size, 2 * 15 + size:2 * 15 + 2 * size] = img_pred
    # Text labels
    cv2.putText(img, 'Noised:', (15, 30), cv2.QT_FONT_NORMAL, 0.8, (1., 1., 1.), 1)
    cv2.putText(img, 'Denoised:', (2 * 15 + size, 30), cv2.QT_FONT_NORMAL, 0.8, (1., 1., 1.), 1)
    # Show
    cv2.imshow('Autoencoder', img)
    cv2.waitKey(0)


def inference(mode, path=None):
    global model
    model.eval()

    print('Running inference...')

    if mode == 'pic':
        with torch.no_grad():
            # Read as grayscale image
            img_noise = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            # Transform using transform_y cuz it's already noised
            img_noise = transform_y(img_noise).to(device)

            # Add batch dimension
            img_noise.unsqueeze_(0)

            # Pass to the model
            img_pred = model(img_noise)

            visualize_denoise(img_noise.numpy().squeeze(), img_pred.numpy().squeeze())

    if mode == 'loader':
        # Lists of noisy and denoised pics
        Images = np.zeros((len(inferset), 28, 28), dtype=np.float)
        Predicts = np.zeros((len(inferset), 28, 28), dtype=np.float)

        pic_ind = 0

        with torch.no_grad():
            # Iterate through batches
            for X, _ in infer_loader:
                # Remember noised pictures
                Images[pic_ind:pic_ind + len(X)] = X.numpy().squeeze(1)

                X = X.to(device)

                # Get and remember denoised pics
                predictions = model(X)
                Predicts[pic_ind:pic_ind + len(X)] = predictions.numpy().squeeze(1)

                pic_ind += len(X)

        for img_noise, img_pred in zip(Images, Predicts):
            visualize_denoise(img_noise, img_pred)


if __name__ == '__main__':
    global src
    src = 'load' # 'load' or 'create'

    read_dataset()

    if src == 'create':
        print('Training a model:')
        create_model()
        train()
        save_model('net')

    if src == 'load':
        print('The model\'s loaded')
        load_model('net')

    # Inference on a local image
    inference('pic', 'example.png')
    # Inference on several images from loader
    inference('loader')
