'''
# Date: 1.7.2020
# Author: sseo
# Descriptor
- 1.8. 트레인시작
- 1.9
'''
import random
import os
import time
import math

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import torch
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt

from preprocess import over_sampling

class GargLeNet(nn.Module):
    __constants__ = ['transform_input']

    def __init__(self, num_classes= 36, transform_input=False, init_weights=True, blocks=None):
        super(GargLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception]
        assert len(blocks) == 2

        conv_block = blocks[0]
        inception_block = blocks[1]

        self.transform_input = transform_input

        # Stem
        self.conv1 = conv_block(3, 64, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        #in_channels, ch1x1, ch3x3red_a, ch3x3_a, ch3x3red_b, ch3x3_b1, ch3x3_b2, ch3x3_pool_proj
        # Inception #1
        self.inception2a1 = inception_block(64, 32, 24, 32, 32, 48, 48, 16)
        self.inception2b1 = inception_block(128, 64, 48, 64, 64, 96, 96, 64)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception #2
        self.inception3b1 = inception_block(288, 64, 48, 64, 64, 96, 96, 32)
        self.inception3b2 = inception_block(256, 64, 48, 64, 64, 96, 96, 32)
        self.inception3b3 = inception_block(256, 64, 48, 64, 64, 96, 96, 32)
        self.inception3b4 = inception_block(256, 64, 48, 64, 64, 96, 96, 32)
        self.inception3c1 = inception_block(256, 128, 96, 128, 128, 192, 192, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception #3
        self.inception4c1 = inception_block(512, 128, 96, 128, 128, 192, 192, 64)
        self.inception4c2 = inception_block(512, 128, 96, 128, 128, 192, 192, 64)

        # AvgPool, Dropout, FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):  # 아직 이해는 다 못했지만 일단 사용 가능한것 같아 냅둠.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):  # 이것도 아직 이해를 못했는데, 우리에게 맞게 변형하거나 삭제.
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # Stem (1)
        x = self.conv1(x)
        x = self.maxpool1(x)

        # Inception #1 (2)
        x = self.inception2a1(x)
        x = self.inception2b1(x)
        x = self.maxpool2(x)

        # Inception #2 (3)
        x = self.inception3b1(x)
        x = self.inception3b2(x)
        x = self.inception3b3(x)
        x = self.inception3b4(x)
        x = self.inception3c1(x)
        x = self.maxpool3(x)

        # Inception #3 (4)
        x = self.inception4c1(x)
        x = self.inception4c2(x)

        # AvgPool, Dropout, and FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self._forward(x)
        return x


# Inception module은 논문에서와 같이 변형. (Inception v1 -> factorized ver.)
class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red_a, ch3x3_a, ch3x3red_b, ch3x3_b1, ch3x3_b2, ch3x3_pool_proj, conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red_a, kernel_size=1),
            conv_block(ch3x3red_a, ch3x3_a, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch3x3red_b, kernel_size=1),
            conv_block(ch3x3red_b, ch3x3_b1, kernel_size=3, padding=1),
            conv_block(ch3x3_b1, ch3x3_b2, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, ch3x3_pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class FinalDataset(Dataset):

    def __init__(self, root_dir, train = True, transform = None, augment = None):
        self.folder_dataset = dset.ImageFolder(root = root_dir)
        self.train = train
        self.transform = transform
        self.augment = augment


    def __getitem__(self,index):

        img_dir, label = self.folder_dataset.imgs[index] #label 은 폴더 index로 리턴
        img = Image.open(img_dir).convert('RGB') # convert to grayscale

        if self.train is True and self.augment is not None:
            augment = np.random.choice(self.augment, 1).tolist()
            augment += self.transform
            # print(len(augment))
        else:
          augment = self.transform

        if self.transform is not None:
            img = transforms.Compose(augment)(img)

        return img, label


    def __len__(self):
        return len(self.folder_dataset.imgs)

def train(model, train_loader, optimizer, criterion, epoch, time_start):
    model.train()
    percent_prev = -1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
#         print(output.shape, target.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #if batch_idx % log_interval == 0:
        percent_curr = 100 * batch_idx // len(train_loader)
        if percent_curr > percent_prev:
            percent_prev = percent_curr
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\tTime duration: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), timeSince(time_start)))
            #torch.save(model.state_dict(),"drive/My Drive/public/results/mnist_cnn.pt")
    return loss.item()
def test(model,  test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("output, target = ")
            print(pred, target)
    test_loss /= len(test_loader.dataset)

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

# From Prof's CNN_MNIST practice code
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


train_dir = "./train_over"
valid_dir = "./val"
output_dir = "./output"

try:
  os.makedirs(output_dir + "/output", exist_ok = True)
except OSError as e:
  if os.path.isdir('.output'): pass
  else:
    print('\nPlease make a directory ./output\n', e)


option = {'train_dir': train_dir,
          'valid_dir': valid_dir,
          'output' : output_dir,
          'input_size': (224,224),
          'batch': 8,
          'epoch': 20,
          'lr': 0.001,
          'momentum': 0.9,
          'log_interval': 2,
          'valid_interval': 2,
          'n_cpu': 16,
          'augment': True,
          'ver': 0.7}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': option['n_cpu'], 'pin_memory': True} if use_cuda else {}

print('option:', option)
print('use cuda:', use_cuda)

if __name__ == '__main__':
    over_sampling('./train_over')

    transform = [transforms.Resize(option['input_size']),
         transforms.ToTensor()]

    augment = [transforms.RandomRotation((-10, 10)),
                  transforms.ColorJitter(),
                  transforms.RandomHorizontalFlip()]

    train_set = FinalDataset(root_dir = option['train_dir'], train = True,
                                 transform = transform, augment = augment if option['augment'] else None)

    train_loader = DataLoader(train_set,
                              shuffle = True,
                              batch_size = option['batch'],
                              **kwargs)

    valid_set = FinalDataset(root_dir = option['valid_dir'], train = False,
                                transform = transform)

    valid_loader = DataLoader(valid_set,
                             shuffle = False,
                             batch_size = 100, # test batch: 100
                             **kwargs)

    model = GargLeNet().to(device)
    #model.load_state_dict(torch.load(output_dir+"/gargle_4.pth"), strict = False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= option['lr'], momentum= option['momentum'])

    epoch_log, train_log, test_log = [], [], []
    for epoch in range(1, option['epoch'] + 1):
        start = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, epoch, start)
        train_log.append(train_loss)

        if epoch % option['valid_interval'] == 0:
            test_loss = test(model, valid_loader)

            epoch_log.append(epoch); test_log.append(test_loss)
            torch.save(model.state_dict(), output_dir + "/gargle{}_{}.pth".format(option['ver'],epoch))

    plt.plot(epoch_log, test_log)
    plt.title('test loss: {}'.format(option['epoch'])); plt.ylabel('loss')
    plt.savefig(output_dir + "/test_{}.png".format(option['ver']), dpi = 300)

    epoch_log = np.arange(1, option['epoch'] + 1)
    plt.plot(epoch_log, train_log)
    plt.title('train loss: {}'.format(option['epoch'])); plt.ylabel('loss')
    plt.savefig(output_dir + "/train_{}.png".format(option['ver']), dpi = 300)
