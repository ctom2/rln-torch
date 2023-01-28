import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GELU(x):
    return nn.GELU()(x)

class RLN(nn.Module):
    def __init__(self):
        super(RLN, self).__init__()

        # h1 branch
        self.h1_avg_pool = torch.nn.AvgPool3d([2,2,2], stride=[2,2,2])
        self.h1_conv1 = torch.nn.Conv3d(in_channels=2, out_channels=4, kernel_size=[3,3,3], stride=[1,1,1], padding='same', bias=False)
        self.h1_bn1 = torch.nn.BatchNorm3d(4)
        self.h1_conv2 = torch.nn.Conv3d(in_channels=4, out_channels=4, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h1_bn2 = torch.nn.BatchNorm3d(4)
        self.h1_conv3 = torch.nn.Conv3d(in_channels=8, out_channels=4, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h1_bn3 = torch.nn.BatchNorm3d(4)

        self.h1_bnA = torch.nn.BatchNorm3d(2)
        
        self.h1_conv4 = torch.nn.Conv3d(in_channels=2, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h1_bn4 = torch.nn.BatchNorm3d(8)
        self.h1_conv5 = torch.nn.Conv3d(in_channels=8, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h1_bn5 = torch.nn.BatchNorm3d(8)
        self.h1_conv6 = torch.nn.Conv3d(in_channels=16, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h1_bn6 = torch.nn.BatchNorm3d(8)

        self.h1_convup = torch.nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=[2,2,2], stride=[2,2,2])
        self.h1_bnB = torch.nn.BatchNorm3d(4)
        
        self.h1_conv7 = torch.nn.Conv3d(in_channels=4, out_channels=4, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h1_bn7 = torch.nn.BatchNorm3d(4)

        self.h1_bnC = torch.nn.BatchNorm3d(1)


        # h2 branch
        self.h2_conv1 = torch.nn.Conv3d(in_channels=2, out_channels=4, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h2_bn1 = torch.nn.BatchNorm3d(4)
        self.h2_conv2 = torch.nn.Conv3d(in_channels=4, out_channels=4, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h2_bn2 = torch.nn.BatchNorm3d(4)
        
        self.h2_bnA = torch.nn.BatchNorm3d(2)

        self.h2_conv3 = torch.nn.Conv3d(in_channels=2, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h2_bn3 = torch.nn.BatchNorm3d(8)
        self.h2_conv4 = torch.nn.Conv3d(in_channels=8, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h2_bn4 = torch.nn.BatchNorm3d(8)

        self.h2_bnB = torch.nn.BatchNorm3d(1)


        # h3 branch
        self.h3_conv1 = torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h3_conv2 = torch.nn.Conv3d(in_channels=10, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h3_conv3 = torch.nn.Conv3d(in_channels=16, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1], padding='same')
        self.h3_bnA = torch.nn.BatchNorm3d(8)



    def forward_fp1(self, x, x_avg):
        x1 = self.h1_conv1(x)
        x1 = self.h1_bn1(x1)
        x1 = GELU(x1)

        x2 = self.h1_conv2(x1)
        x2 = self.h1_bn2(x2)
        x2 = GELU(x2)

        xx = torch.cat([x1,x2], 1)

        x3 = self.h1_conv3(xx)
        x3 = self.h1_bn3(x3)
        x3 = GELU(x3) + x_avg

        return x3


    def forward_bp1(self, x):
        x4 = self.h1_conv4(x)
        x4 = self.h1_bn4(x4)
        x4 = GELU(x4)

        x5 = self.h1_conv5(x4)
        x5 = self.h1_bn5(x5)
        x5 = GELU(x5)

        xx = torch.cat([x4,x5], 1)

        x6 = self.h1_conv6(xx)
        x6 = self.h1_bn6(x6)
        x6 = GELU(x6)

        xu = self.h1_convup(x6)
        xu = self.h1_bnB(xu)
        xu = GELU(xu)

        x7 = self.h1_conv7(xu)
        x7 = self.h1_bn7(x7)
        x7 = GELU(x7)

        return x7


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


    def forward_h1_branch(self, y, yy):
        x = self.h1_avg_pool(yy)
        xt = GELU(torch.cat([x,x], 1))

        # FP1
        x3 = self.forward_fp1(x, xt)

        # C_AVE
        xy = torch.mean(x3, dim=1, keepdim=True)
        xy = torch.cat([xy,xy], 1)

        # DV1
        xy = (x)/(xy + 0.001)
        xy = self.h1_bnA(xy)

        # BP1 + BP1_up
        x7 = self.forward_bp1(xy)

        # C_AVE
        xz = torch.mean(x7, dim=1, keepdim=True)        

        # E1
        temp = xz * y # <--- use for loss

        xq = self.h1_bnC(temp)
        e1 = GELU(xq)

        return e1, temp


    def forward_h2_branch(self, yy, e1):
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
        yz = e1 * yz

        yz = self.h2_bnB(yz)
        e2 = GELU(yz)

        return e2


    def forward_h3_branch(self, e1, e2):
        z1 = self.h3_conv1(e2)
        z1 = GELU(z1)
        zz1 = torch.cat([z1,e2,e1], 1)

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

        # ******** h1 branch ********
        e1, temp = self.forward_h1_branch(y, yy)

        # ******** h2 branch ********
        e2 = self.forward_h2_branch(yy, e1)

        # ******** h3 branch ********
        ef = self.forward_h3_branch(e1, e2)

        return ef, temp