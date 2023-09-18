import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

i=1
class UNet(nn.Module):

    def __init__(self, n_channels, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(n_channels, 64*i)
        self.dconv_down2 = double_conv(64*i, 128*i)
        self.dconv_down3 = double_conv(128*i, 256*i)
        self.dconv_down4 = double_conv(256*i, 512*i)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256*i + 512*i, 256*i)
        self.dconv_up2 = double_conv(128*i + 256*i, 128*i)
        self.dconv_up1 = double_conv(128*i + 64*i, 64*i)
        
        self.conv_last = nn.Conv2d(64*i, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)  
               
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out