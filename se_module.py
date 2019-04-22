# On 05-02-2019
# For aifr
# By Saurav Rai
from torch import nn

"""
Aggregate block:
uses: sum as the aggregation operation
      aggregation factor specifies the number of channels in each group
      #out_feature_maps = #in_feature_maps / k.
"""

class Aggregate:
    def __init__(self, aggregate_factor):
        self.aggregate_factor= aggregate_factor
    
    def aggregate(self, x):
        b, c, h, w = x.size()
        res= x.reshape(b, c//self.aggregate_factor, self.aggregate_factor, h, w).sum(2)
        return res

class SELayer(nn.Module):
    def __init__(self, channel, reduction= 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        #print("\n______SELAYER____\n")
        #print('x size',x.size()) 
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c) #Squueze step

        y = self.fc(y).view(b, c, 1, 1) #Excitation step(1)
        y= x * y #Excitation step(2)
        return y
