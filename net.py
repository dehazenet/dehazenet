import torch
import torch.nn as nn

class DehazeNet(nn.Module):
    def __init__(self, input=16, groups=4):
        super(DehazeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        self.relu4 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6)
        
    
    def forward(self, x):
        #feature extraction
        out = self.conv1(x)
        out = self.relu1(out)
        #maxout
        max_1 = torch.max(out[:,0:4,:,:],out[:,4:8,:,:])
        max_2 = torch.max(out[:,8:12,:,:],out[:,12:16,:,:])
        out = torch.max(max_1,max_2)

        #multi-scale Mapping
        out1 = self.conv2(out)
        out1 = self.relu2(out1)
        out2 = self.conv3(out)
        out2 = self.relu3(out2)
        out3 = self.conv4(out)
        out3 = self.relu4(out3)
        y = torch.cat((out1,out2,out3), dim=1)
        #Local Extremum
        y = self.maxpool(y)
        #non-linear Regression
        y = self.conv5(y)
        y = torch.max(y, torch.zeros(y.shape[0],y.shape[1],y.shape[2],y.shape[3]).cuda())
        y = torch.min(y, torch.ones(y.shape[0],y.shape[1],y.shape[2],y.shape[3]).cuda())
        return y