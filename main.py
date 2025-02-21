from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#import focal_loss
#from focal_loss import FocalLoss


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--checkpoint', type=str, default=None, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")

args = parser.parse_args()



#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data Initialization and Loading
from data import initialize_data, data_transforms, train_transforms, jitter_hue,jitter_brightness,jitter_saturation,jitter_contrast,rotate,hvflip,shear,translate,center,hflip,vflip,grayscale # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
   torch.utils.data.ConcatDataset([datasets.ImageFolder(args.data + '/train_images',
   transform=data_transforms),
   datasets.ImageFolder(args.data + '/train_images',
   transform=grayscale),datasets.ImageFolder(args.data + '/train_images',
   transform=jitter_brightness),datasets.ImageFolder(args.data + '/train_images',
   transform=jitter_saturation),datasets.ImageFolder(args.data + '/train_images',
   transform=jitter_hue),datasets.ImageFolder(args.data + '/train_images',
   transform=jitter_contrast),datasets.ImageFolder(args.data + '/train_images',
   transform=translate),datasets.ImageFolder(args.data + '/train_images',
   transform=center),datasets.ImageFolder(args.data + '/train_images',
   transform=grayscale),datasets.ImageFolder(args.data + '/train_images',
   transform=rotate),datasets.ImageFolder(args.data + '/train_images',
   transform=hvflip),datasets.ImageFolder(args.data + '/train_images',
   transform=shear)]), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

#train_loader = torch.utils.data.DataLoader(
#    datasets.ImageFolder(args.data + '/train_images',
#                         transform=train_transforms),
#    batch_size=args.batch_size, shuffle=True, num_workers=4)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4)

print(args.lr)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

from model import Net, SpatialNet
#model = Net()
model = SpatialNet()

# from conv_net import ConvNet
# model = ConvNet()

#from model import VGG, VGG19
#model = VGG(VGG19)

#from googLeNet import GoogLeNet
#model = GoogLeNet()


#from efficient import EfficientNet, EfficientNetB0
#model = EfficientNetB0()

if args.checkpoint is not None:
    print("using checkpoint")
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

#print(model)

model = model.cuda()
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=3,factor=0.1,verbose=True)
#loss = FocalLoss(class_num = 43, gamma=1.5, size_average = False)


def train(epoch):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_num = batch_idx+1
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss = F.cross_entropy(output, target).cuda()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_num % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_num * len(data), len(train_loader.dataset),
                100. * batch_num / len(train_loader), loss.item()))
    return 100. * correct / len(train_loader.dataset) ,train_loss / len(train_loader.dataset)

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        validation_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss,2))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return 100. * correct / len(val_loader.dataset), validation_loss

# step=10
tran_arr=[]
val_arr=[]
tran_acc_arr=[]
val_acc_arr=[]
for epoch in range(1, args.epochs + 1):
    tran_acc, tran_loss = train(epoch)
    val_acc, val_loss = validation()
    # if epoch % step :
    # print("train: " , tran_loss)
    # print("val:" , val)
    tran_arr.append(tran_loss)
    val_arr.append(val_loss)
    tran_acc_arr.append(tran_acc)
    val_acc_arr.append(val_acc)
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')


plt.plot(tran_arr, c='b', label='Training Loss')
plt.plot(val_arr, c='r', label='Validation Loss')
plt.legend()
plt.savefig('loss.png')
plt.figure()
plt.plot(tran_acc_arr, c='b', label='Training Accuracy')
plt.plot(val_acc_arr, c='r', label='Validation Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.figure()
plt.plot(val_acc_arr, c='r', label='Validation Accuracy')
plt.legend()
plt.savefig('validation_accuracy.png')