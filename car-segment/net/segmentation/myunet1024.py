# unet from scratch

#from common import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view(-1)
        t = labels.view(-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum() / w.sum()
        return loss

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss,self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num, -1)
        w2    = w*w
        m1    = (probs).view(num, -1)
        m2    = (labels).view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum() / num
        return score

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
        #nn.ReLU(inplace=True),
        nn.PReLU()
    ]
# based on https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py



class UNet1024 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet1024, self).__init__()
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
        self.down7 = nn.Sequential(
            *make_conv_bn_relu(512,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        self.center = nn.Sequential(
            *make_conv_bn_relu(1024, 2048, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(2048,2048, kernel_size=3, stride=1, padding=1 ),
        )

        self.up7 = nn.Sequential(
            *make_conv_bn_relu(1024+2048,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     1024,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     1024,1024, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
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

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #1024
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #512

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #256

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #128

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #64

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #32

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 16
	
        down7 = self.down7(out)
        out   = F.max_pool2d(down7, kernel_size=2, stride=2) # 8

        out   = self.center(out)
	
        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down7, out],1)
        out   = self.up7(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #512
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #1024
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        #out   = F.upsample_bilinear(out, scale_factor=2) #1024
        #x     = F.upsample_bilinear(x,   scale_factor=2)
        #out   = torch.cat([x, out],1)
        #out   = self.up0(out)


        out   = self.classify(out)

        return out






class UNet1024_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet1024_2, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8
        self.down7 = nn.Sequential(
            *make_conv_bn_relu(512,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        self.center = nn.Sequential(
            *make_conv_bn_relu(1024, 2048, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(2048,2048, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(2048,2048, kernel_size=3, stride=1, padding=1 ),
        )

        self.up7 = nn.Sequential(
            *make_conv_bn_relu(1024+2048,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     1024,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     1024,1024, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
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

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #1024
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #512

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #256

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #128

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #64

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #32

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 16
	
        down7 = self.down7(out)
        out   = F.max_pool2d(down7, kernel_size=2, stride=2) # 8

        out   = self.center(out)
	
        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down7, out],1)
        out   = self.up7(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #512
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #1024
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        #out   = F.upsample_bilinear(out, scale_factor=2) #1024
        #x     = F.upsample_bilinear(x,   scale_factor=2)
        #out   = torch.cat([x, out],1)
        #out   = self.up0(out)


        out   = self.classify(out)

        return out





class UNet1024_3 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet1024_3, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_3 ------------------------------------------------------------------------
        #256
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8
        
        self.down7 = nn.Sequential(
            *make_conv_bn_relu(512,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        self.center = nn.Sequential(
            *make_conv_bn_relu(1024, 2048, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(2048,2048, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(2048,2048, kernel_size=3, stride=1, padding=1 ),
        )
        self.deconv7 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up7 = nn.Sequential(
            *make_conv_bn_relu(1024+2048,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     1024,1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     1024,1024, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16
        self.deconv6 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16
        self.deconv5 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32
        self.deconv4 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32+3,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #1024
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #512

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #256

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #128

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #64

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #32

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 16
	
        down7 = self.down7(out)
        out   = F.max_pool2d(down7, kernel_size=2, stride=2) # 8

        out   = self.center(out)
	
        out   = self.deconv7(out) #16
        out   = torch.cat([down7, out],1)
        out   = self.up7(out)

        out   = self.deconv6(out) #32
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = self.deconv5(out) #64
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = self.deconv4(out) #128
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = self.deconv3(out) #256
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = self.deconv2(out) #512
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = self.deconv1(out) #1024
        out   = torch.cat([down1, out, x],1)
        out   = self.up1(out)

        #out   = F.upsample_bilinear(out, scale_factor=2) #1024
        #x     = F.upsample_bilinear(x,   scale_factor=2)
        #out   = torch.cat([x, out],1)
        #out   = self.up0(out)


        out   = self.classify(out)

        return out



import os
from torch.autograd import Variable

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    batch_size  = 2
    C,H,W = 3,1024,1024

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

        net = UNet1024_3(in_shape=(C,H,W), num_classes=1).cuda().train()
        x = Variable(inputs).cuda()
        y = Variable(labels).cuda()
        logits = net.forward(x)

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        net.cuda()
        print(net)

        print('logits')
        print(logits)
    #input('Press ENTER to continue.')
