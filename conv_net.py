import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 128, padding=(2,2), kernel_size=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(128)
        # self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(128, 256, padding=(2,2), kernel_size=5)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(512)
        # self.drop2 = nn.Dropout2d(0.2)
        
        self.conv5 = nn.Conv2d(256, 512,  padding=(2,2), kernel_size=5)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3)
        self.norm5 = nn.BatchNorm2d(512)
        self.norm6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, nclasses)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
   
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    # Spatial transformer network forward function
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
        x = self.norm1(x)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = F.max_pool2d(x,2)
        # x = F.dropout2d(x, p=0.2, training=self.training)
        #print(x.shape)

        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        x = F.max_pool2d(x,2)
        # x = F.dropout2d(x, p=0.2, training=self.training)
        #print(x.shape)

        x = F.relu(self.conv5(x))
        x = self.norm5(x)
        x = F.relu(self.conv6(x))
        x = self.norm6(x)
        # x = F.max_pool2d(x,2)
        x = F.dropout2d(x, p=0.5, training=self.training)
        
        x = x.view(-1, 512*6*6)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x