import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GELU(x):
    return nn.GELU()(x)

class RLN_2(nn.Module):
    def __init__(self):
        super(RLN_2, self).__init__()

        # h2 branch
        self.h2_conv1 = torch.nn.Conv3d(in_channels=2, out_channels=8, kernel_size=5, stride=[1,1,1], padding='same')
        self.h2_bn1 = torch.nn.BatchNorm3d(8)
        self.h2_conv2 = torch.nn.Conv3d(in_channels=8, out_channels=8, kernel_size=5, stride=[1,1,1], padding='same')
        self.h2_bn2 = torch.nn.BatchNorm3d(8)
        
        self.h2_bnA = torch.nn.BatchNorm3d(2)

        self.h2_conv3 = torch.nn.Conv3d(in_channels=2, out_channels=16, kernel_size=5, stride=[1,1,1], padding='same')
        self.h2_bn3 = torch.nn.BatchNorm3d(16)
        self.h2_conv4 = torch.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=5, stride=[1,1,1], padding='same')
        self.h2_bn4 = torch.nn.BatchNorm3d(16)

        self.h2_bnB = torch.nn.BatchNorm3d(1)


        # h3 branch
        self.h3_conv1 = torch.nn.Conv3d(in_channels=1, out_channels=16, kernel_size=5, stride=[1,1,1], padding='same')
        self.h3_conv2 = torch.nn.Conv3d(in_channels=17, out_channels=16, kernel_size=5, stride=[1,1,1], padding='same')
        self.h3_conv3 = torch.nn.Conv3d(in_channels=32, out_channels=16, kernel_size=5, stride=[1,1,1], padding='same')
        self.h3_bnA = torch.nn.BatchNorm3d(16)



    def forward_fp2(self, x):
        y1 = self.h2_conv1(x)
        y1 = self.h2_bn1(y1)
        y1 = GELU(y1)

        y2 = self.h2_conv2(y1)
        y2 = self.h2_bn2(y2)
        y2 = GELU(y2)

        return y2


    def forward_bp2(self, x):
        y3 = self.h2_conv3(x)
        y3 = self.h2_bn3(y3)
        y3 = GELU(y3)

        y4 = self.h2_conv4(y3)
        y4 = self.h2_bn4(y4)
        y4 = GELU(y4) + torch.ones(y4.shape).to(device)

        return y4


    def forward_h2_branch(self, yy, y):
        y2 = self.forward_fp2(yy)

        # C_AVE
        yq = torch.mean(y2, dim=1, keepdim=True)
        yq = torch.cat([yq,yq], 1)

        # DV2
        yq = (yy)/(yq + 0.001)
        yq = self.h2_bnA(yq)
        
        # BP2
        y4 = self.forward_bp2(yq)

        # C_AVE
        yz = torch.mean(y4, dim=1, keepdim=True)

        # E2
        yz = y * yz

        yz = self.h2_bnB(yz)
        e2 = GELU(yz)

        return e2


    def forward_h3_branch(self, e2):
        z1 = self.h3_conv1(e2)
        z1 = GELU(z1)
        zz1 = torch.cat([z1,e2], 1)

        z2 = self.h3_conv2(zz1)
        z2 = GELU(z2)

        m = torch.cat([z1,z2], 1)
        z3 = self.h3_conv3(m)
        z3 = self.h3_bnA(z3)
        z3 = GELU(z3)

        ef = torch.mean(z3, dim=1, keepdim=True)

        return ef


    def forward(self, y):
        yy = torch.cat([y,y],1)

        # ******** h2 branch ********
        e2 = self.forward_h2_branch(yy, y)

        # ******** h3 branch ********
        ef = self.forward_h3_branch(e2)

        return ef