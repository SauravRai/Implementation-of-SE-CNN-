#Written by Saurav Rai on 7 feb
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import ResNet
from se_module import SELayer, Aggregate


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class aifrNet(nn.Module):
    def __init__(self, channels, stride=1, reduction=16):
        super(aifrNet, self).__init__()
        self.se = SELayer(channels, reduction)
        self.se_net = SELayer(192, reduction)
        self.agg = Aggregate(4) #ag_fact =4 if we use aggregration also

        self.fc_id =  mfm(5*5*192, 256, type = 0)
        self.fc_output = nn.Linear(256, 10000)

        self.fc_age = mfm(5*5*192,256,type = 0)
        self.fc_age_output = nn.Linear(256, 58)
        self.genmodel = GenModel(192)     
        
        self.fc_layer = nn.Linear(686,4800)
        self.fc_layer1 = nn.Linear(58,4800)
        self.fc_layer2 = nn.Linear(10000,4800)

        '''
        THIS IDEA FROM FIRST PAPER
        self.feat = nn.Sequential(
                            mfm(4800,1024 , type = 0))
        self.fc2 = nn.Linear(4800,1024)
        self.feat1 = nn.Sequential(
                            mfm(1024,1024 , type = 0))

        #THIS fc layer is used for the estimation of the age
        self.fc_age = nn.Linear(5*5*192,58)
        self.fc3 = nn.Linear(1024,1024)
        self.feat2 = nn.Sequential(
                            mfm(1024,1024 , type = 0))
        '''
    def forward(self, x): #x= bsxhxwxc
        #Here x is the original values from the lightcnn 4 model
        original_x  = x 
        
        out1 =  self.se_net(x) 
        out1 = original_x + out1 #First ResNet Block
 
        out2 = self.se_net(out1) 
        out2 = out1 + out2 #Second ResNet Block
         
        out3 = self.se_net(out2) #Third ResNet Block
        out3 = out2 + out3
        out = out3 + original_x 

        out = out.view(-1,192*5*5)
      
        out = self.fc_id(out) 

        embeddings = self.fc_output(out) #Here embeddings is like x 
        
        #return age_featt, identity_featt , embeddings
        return embeddings, embeddings , embeddings


