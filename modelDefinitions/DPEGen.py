import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
from modelDefinitions.basicBlocksDPE import *

class DPENet(nn.Module):
    def __init__(self):
        super(DPENet, self).__init__()

        self.inConv = nn.Conv2d(3,64, 3,1,1)
        self.FM0 = ResidualBlock(64)

        self.FM1 = ResidualBlock(64)
        self.gate1 = GatedConv2d(64, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64,96, kernel_size=3, stride=2, padding=1)

        self.FM2 = ResidualBlock(96)
        self.gate2 = GatedConv2d(96, 96, 3, padding=1)
        self.down2 = nn.Conv2d(96,128, kernel_size=3, stride=2, padding=1)
        
        self.FM3 = ResidualBlock(128)
        self.gate3 = GatedConv2d(128, 128, 3, padding=1)
        self.down3 = nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1)

        self.FM4 = ResidualBlock(160)
        self.gate4 = GatedConv2d(160, 160, 3, padding=1)
        self.down4 = nn.Conv2d(160, 192, kernel_size=3, stride=2, padding=1)


        self.FM5 = ResidualBlock(192)
        self.gate5 = GatedConv2d(192, 192, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(192, 160, 2, 2)

        self.FM6 = ResidualBlock(160)
        self.up2 = nn.ConvTranspose2d(160, 128, 2,2)

        self.FM7 = ResidualBlock(128)
        self.up3 = nn.ConvTranspose2d(128, 96, 2,2)

        self.FM8 = ResidualBlock(96)
        self.up4 = nn.ConvTranspose2d(96, 64, 2,2)
        
        self.FM9 = ResidualBlock(64)

        self.outc = nn.Conv2d(64,3,1,)

    def forward(self, x):

        x1_1 = self.inConv(x)
        #x1_2 = self.FM0(x1_1)  # 

        x2_1 = self.FM1(x1_1)
        g_2 = self.gate1(x2_1)
        x2_2 = F.relu(self.down1(x2_1))

        x3_1 = self.FM2(x2_2)
        g_3 = self.gate2(x3_1)
        x3_2 = F.relu(self.down2(x3_1))

        x4_1 = self.FM3(x3_2)
        g_4 = self.gate3(x4_1)
        x4_2 = F.relu(self.down3(x4_1))

        x5_1 = self.FM4(x4_2)
        g_5 = self.gate4(x5_1)
        x5_2 = F.relu(self.down4(x5_1))

        x6_1 = self.FM5(x5_2)
        x6_2 = F.relu(self.up1(x6_1)) + g_5

        x7_1 = self.FM6(x6_2)
        x7_2 = F.relu(self.up2(x7_1)) + g_4

        x8_1 = self.FM7(x7_2)
        x8_2 = F.relu(self.up3(x8_1)) + g_3

        x9_1 = self.FM8(x8_2)
        x9_2 = F.relu(self.up4(x9_1)) + g_2

        #x10 = self.FM9(x9_2) #
        out = torch.tanh(self.outc(x9_2) + x)

        return out#self.outc(x9_2) + x


#net = Noise_DPN()
#summary(net, input_size = (3, 256, 256))