from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os, sys
import argparse
import numpy as np
from models import *
from utils import progress_bar


def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()

def to_variable(tensor, requires_grad=False):
    tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)

class pathology_dataset(Dataset):
    def __init__(self, csv, img_dir, transform=None):
        self.ids = []
        self.ys = []
        with open(csv) as lines:
            for line in lines:
                id, y = line.strip().split(',')
                self.ids.append(id)
                self.ys.append(y)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, 'image' + str(self.ids[idx]) + '.png')
        x = Image.open(img_name).convert('RGB')
        y = int(self.ys[idx])
        if self.transform:
            x = self.transform(x)
        return to_tensor(x), to_tensor(y).long()

parser = argparse.ArgumentParser(description='PyTorch cifar10 for pathology')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_step', type=int, default=80, help='lr_step')
parser.add_argument('--epoch', type=int, default=200, help='epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sp', type=str, default='0', help='splits')
parser.add_argument('--img_dir', type=str, default='../data/', help='img dir')
parser.add_argument('--id', type=str, default='00', help='model id')
parser.add_argument('--transfer',  action='store_true', help='whether to transfer to new task with a pretrained model')
parser.add_argument('--pretrained',  default='', type=str, help='the pretrained model')
args = parser.parse_args()

fp = open('checkpoint/log_' + str(args.id) + '.txt', 'w')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(64, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = pathology_dataset(os.path.join('../splits', 'sp' + str(args.sp), 'tr_lst'), args.img_dir, transform_train)
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = pathology_dataset(os.path.join('../splits', 'sp' + str(args.sp), 'tt_lst'), args.img_dir, transform_test)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.transfer:
    net = ResNet18(num_classes=10, transfer=True)
else:
    net = ResNet18(num_classes=2)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.transfer:
    # Load checkpoint.
    print('==> Transfer from pretrained model')
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net'])
    print(net.module.linear.weight)
    net.module.linear = nn.Linear(512*4, 2)
    print(net.module.linear.weight)


net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    line = '{},{}'.format(epoch, acc)
    fp.write(line + '\n')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + str(args.id) + '_ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, args.epoch):
    if (epoch + 1) % args.lr_step == 0:
        optim_state = optimizer.state_dict()
        org_lr = optim_state['param_groups'][0]['lr']
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 10
        print('adujust lr from  {} to {}'.format(org_lr, optim_state['param_groups'][0]['lr']))
        optimizer.load_state_dict(optim_state)
    train(epoch)
    test(epoch)
fp.close()
