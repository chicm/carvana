# unet from scratch

#from common import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

#  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
#  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)

#https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)

        #
        # logits_flat  = logits.view (-1)
        # targets_flat = targets.view(-1)
        # return StableBCELoss()(logits_flat,targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score


## -------------------------------------------------------------------------------------

def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        #nn.PReLU()
    ]


class UNet128 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet128, self).__init__()
        in_channels, height, width = in_shape

        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=1, stride=1, padding=0 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=1, stride=1, padding=0 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #128

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)
        out   = self.classify(out)
        #out   = F.sigmoid(out)

        return out



# a bigger version for 256x256
class UNet256 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256, self).__init__()
        in_channels, height, width = in_shape

        #256

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=1, stride=2, padding=0 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=1, stride=1, padding=0 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #256

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = self.up0(out)

        out   = self.classify(out)

        return out





# a bigger version for 256x256
class UNet256_1 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256_1, self).__init__()
        in_channels, height, width = in_shape

        #256

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=2, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #256

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = self.up0(out)

        out   = self.classify(out)

        return out






class UNet128_1 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet128_1, self).__init__()
        in_channels, height, width = in_shape

        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = self.classify(out)

        return out



# based on https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

class UNet128_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet128_2, self).__init__()
        in_channels, height, width = in_shape

        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = self.classify(out)

        return out


class UNet256_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256_2, self).__init__()
        in_channels, height, width = in_shape

        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #256
        down0 = self.down0(x)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)

        out   = self.classify(out)

        return out



class UNet512_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet512_2, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down0a = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up0a = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down0a = self.down0a(x)
        out    = F.max_pool2d(down0a, kernel_size=2, stride=2) #64

        down0 = self.down0(out)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down0a, out],1)
        out   = self.up0a(out)

        out   = self.classify(out)

        return out




class UNet_double_1024_5 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet_double_1024_5, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(  3+16,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #64

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #64

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #32

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #16

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #1024
        x     = F.upsample_bilinear(x,   scale_factor=2)
        out   = torch.cat([x, out],1)
        out   = self.up0(out)


        out   = self.classify(out)

        return out



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4, drop_rate=0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

        self.of = num_input_features + num_layers * growth_rate

class _TransitionDown(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        self.of = num_output_features

class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionUp, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('deconv', nn.ConvTranspose2d(num_input_features, num_output_features, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.of = num_output_features

class DenseUnet1(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(DenseUnet1, self).__init__()
        in_channels, height, width = in_shape
        self.block1 = _DenseBlock(4, 3, 6)
        self.down1 = _TransitionDown(self.block1.of, self.block1.of // 2 )
        
        self.block2 = _DenseBlock(8, self.down1.of, 6)
        self.down2 = _TransitionDown(self.block2.of, self.block2.of // 2)
        
        self.block3 = _DenseBlock(8, self.down2.of, 6)
        self.down3 = _TransitionDown(self.block3.of, self.block3.of // 2)
        
        self.block4 = _DenseBlock(8, self.down3.of, 6)
        self.down4 = _TransitionDown(self.block4.of, self.block4.of // 2)

        self.block5 = _DenseBlock(8, self.down4.of, 6)
        self.down5 = _TransitionDown(self.block5.of, self.block4.of // 2)

        self.block6 = _DenseBlock(8, self.down5.of, 6)
        self.down6 = _TransitionDown(self.block6.of, self.block4.of // 2)

        self.center = _DenseBlock(8, self.down6.of, 6)

        self.up6 = _TransitionUp(self.center.of, self.down6.of // 2)
        self.ublock6 = _DenseBlock(8, self.up6.of+self.block6.of, 3)

        self.up5 = _TransitionUp(self.ublock6.of, self.down5.of // 2)
        self.ublock5 = _DenseBlock(8, self.up5.of+self.block5.of, 3)

        self.up4 = _TransitionUp(self.ublock5.of, self.down4.of // 2)
        self.ublock4 = _DenseBlock(8, self.up4.of+self.block4.of, 3)

        self.up3 = _TransitionUp(self.ublock4.of, self.down3.of // 2)
        self.ublock3 = _DenseBlock(8, self.up3.of+self.block3.of, 3)

        self.up2 = _TransitionUp(self.ublock3.of, self.down2.of // 2)
        self.ublock2 = _DenseBlock(8, self.up2.of+self.block2.of, 3)

        self.up1 = _TransitionUp(self.ublock2.of, self.down1.of // 2)
        self.ublock1 = _DenseBlock(8, self.up1.of+self.block1.of, 3)

        self.classifier = nn.Conv2d(self.ublock1.of, num_classes, kernel_size=1, stride=1, padding=0 )

    def forward(self, x):
        db1 = self.block1(x)
        out = self.down1(db1)
        #print(out.size())
        
        db2 = self.block2(out)
        out = self.down2(db2)
        #print(out.size())
        
        db3 = self.block3(out)
        out = self.down3(db3)
        #print(out.size())
        
        db4 = self.block4(out)
        out = self.down4(db4)
        #print(out.size())
        
        db5 = self.block5(out)
        out = self.down5(db5)
        #print(out.size())
        
        db6 = self.block6(out)
        out = self.down6(db6)
        #print(out.size())

        out = self.center(out)

        out = self.up6(out)
        #print(out.size())
        out = torch.cat([out, db6], 1)
        out = self.ublock6(out)
        #print(out.size())

        out = self.up5(out)
        #print(out.size())
        out = torch.cat([out, db5], 1)
        out = self.ublock5(out)
        #print(out.size())

        out = self.up4(out)
        #print(out.size())
        out = torch.cat([out, db4], 1)
        out = self.ublock4(out)
        #print(out.size())

        out = self.up3(out)
        #print(out.size())
        out = torch.cat([out, db3], 1)
        out = self.ublock3(out)
        #print(out.size())
        
        out = self.up2(out)
        #print(out.size())
        out = torch.cat([out, db2], 1)
        out = self.ublock2(out)
        #print(out.size())

        out = self.up1(out)
        #print(out.size())
        out = torch.cat([out, db1], 1)
        out = self.ublock1(out)
        #print(out.size())

        out = self.classifier(out)
        #print(out.size())

        return out

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = Pool2DLayer(l, 2, mode='max')

    return l
    # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """

    # Upsample
    l = ConcatLayer(block_to_upsample)
    l = Deconv2DLayer(l, n_filters_keep, filter_size=3, stride=2,
                      crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
    # Concatenate with skip connection
    l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution



class UNet_double_1024_6 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet_double_1024_6, self).__init__()
        in_channels, height, width = in_shape

        #1024
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=1 ),
        )
        #512

        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(8, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(  3+8+16,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):
        
        down0 = self.down0(x) #1024
        out  = F.max_pool2d(down0, kernel_size=2, stride=2) #512
        #512
        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #256

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #128

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #64

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #32

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #16

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 8

        out   = self.center(out) # 8
        #print(out.size())

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #512
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #1024
        #x     = F.upsample_bilinear(x,   scale_factor=2)
        out   = torch.cat([x, out, down0],1)
        out   = self.up0(out)


        out   = self.classify(out)

        return out








from torch.autograd import Variable

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    batch_size  = 4
    C,H,W = 3,512,512

    # if 0: # CrossEntropyLoss2d()
    #     inputs = torch.randn(batch_size,C,H,W)
    #     labels = torch.LongTensor(batch_size,H,W).random_(1)
    #
    #     net = UNet512_2(in_shape=(C,H,W), num_classes=2).cuda().train()
    #     x = Variable(inputs).cuda()
    #     y = Variable(labels).cuda()
    #     logits = net.forward(x)
    #
    #     loss = CrossEntropyLoss2d()(logits, y)
    #     loss.backward()
    #
    #     print(type(net))
    #     print(net)
    #
    #     print('logits')
    #     print(logits)



    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        net = DenseUnet1(in_shape=(C,H,W), num_classes=1).cuda().train()
        x = Variable(inputs).cuda()
        y = Variable(labels).cuda()
        logits = net.forward(x)

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        print(net)

        print('logits')
        print(logits)
    #input('Press ENTER to continue.')
