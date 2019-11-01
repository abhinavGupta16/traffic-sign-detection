import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, padding=(2,2), kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, padding=(1,1), kernel_size=3)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(32)
        # self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(32, 64, padding=(2,2), kernel_size=5)
        self.conv4 = nn.Conv2d(64, 64, padding=(1,1), kernel_size=3)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(64)
        # self.drop2 = nn.Dropout2d(0.2)
        
        self.conv5 = nn.Conv2d(64, 128, padding=(2,2), kernel_size=5)
        self.conv6 = nn.Conv2d(128, 128, padding=(1,1), kernel_size=3)
        self.norm5 = nn.BatchNorm2d(128)
        self.norm6 = nn.BatchNorm2d(128)
        # self.drop3 = nn.Dropout2d(0.2)
        
        self.conv7 = nn.Conv2d(128, 256, padding=(2,2), kernel_size=5)
        self.conv8 = nn.Conv2d(256, 256, padding=(1,1), kernel_size=3)
        self.norm7 = nn.BatchNorm2d(256)
        self.norm8 = nn.BatchNorm2d(256)
        
        self.conv9 = nn.Conv2d(256, 512, padding=(2,2), kernel_size=5)
        self.conv10 = nn.Conv2d(512, 512, padding=(1,1), kernel_size=3)
        self.norm9 = nn.BatchNorm2d(512)
        self.norm10 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, nclasses)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.norm1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.norm2(x)
        x = F.max_pool2d(x,2)
        x = F.dropout2d(x, p=0.2, training=self.training)
        #print(x.shape)

        x = F.leaky_relu(self.conv3(x))
        x = self.norm3(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.norm4(x)
        #x = F.max_pool2d(x,2)
        x = F.dropout2d(x, p=0.2, training=self.training)
        #print(x.shape)

        x = F.leaky_relu(self.conv5(x))
        x = self.norm5(x)
        x = F.leaky_relu(self.conv6(x))
        x = self.norm6(x)
        x = F.max_pool2d(x,2)
        x = F.dropout2d(x, p=0.2, training=self.training)
        #print(x.shape)        

        x = F.leaky_relu(self.conv7(x))
        x = self.norm7(x)
        x = F.leaky_relu(self.conv8(x))
        x = self.norm8(x)
        #x = F.max_pool2d(x,2)
        x = F.dropout2d(x, p=0.2, training=self.training)
        #print(x.shape)

        x = F.leaky_relu(self.conv9(x))
        x = self.norm9(x)
        x = F.leaky_relu(self.conv10(x))
        x = self.norm10(x)
        x = F.max_pool2d(x,2)
        x = F.dropout2d(x, p=0.2, training=self.training)
        
        #print(x.shape)
        x = x.view(x.size(0),-1)
        #print(x.shape)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NNNet(nn.Module):

    def __init__(self):
        super(NNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, padding=(2,2), kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1   = nn.Linear(32*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, nclasses)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        #out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return  F.log_softmax(out,dim=1)

VGG11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
VGG19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_name)
        self.classifier = nn.Linear(512, nclasses)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class SpatialNet(nn.Module):
    def __init__(self):
        super(SpatialNet, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.norm1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.conv2 = nn.Conv2d(150, 150, padding=(1,1), kernel_size=3)
        self.norm2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.conv3 = nn.Conv2d(250, 250, padding=(1,1), kernel_size=3)
        self.norm3 = nn.BatchNorm2d(250)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(250*2*2, 350)
        self.fc2 = nn.Linear(350, nclasses)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
   
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    # STN forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = self.norm1(x)
        x = self.dropout(x)

        # CNN Layer
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = self.norm2(x)
        x = self.dropout(x)
        
        #CNN Layer
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = self.norm3(x)
        x = self.dropout(x)
        
        x = x.view(-1, 250*2*2)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
