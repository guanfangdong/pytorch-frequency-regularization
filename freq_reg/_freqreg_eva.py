import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from ._dct import idct_4d



def getZIdx_4d(data):
     
    zmat = torch.empty(data.shape)


    maxval1 = data.shape[0] + 1.0
    maxval2 = data.shape[1] + 1.0
    maxval3 = data.shape[2] + 1.0
    maxval4 = data.shape[3] + 1.0


    matdim1 = torch.tensor(range(data.shape[0])) + torch.tensor(range(data.shape[0]))/maxval1
    matdim1 = matdim1.expand(data.shape[1], data.shape[2], data.shape[3], data.shape[0])

    matdim2 = torch.tensor(range(data.shape[1])) + torch.tensor(range(data.shape[1]))/(maxval1*maxval2)
    matdim2 = matdim2.expand(data.shape[0], data.shape[2], data.shape[3], data.shape[1])

    matdim3 = torch.tensor(range(data.shape[2])) + torch.tensor(range(data.shape[2]))/(maxval1*maxval2*maxval3)
    matdim3 = matdim3.expand(data.shape[0], data.shape[1], data.shape[3], data.shape[2])

    matdim4 = torch.tensor(range(data.shape[3])) + torch.tensor(range(data.shape[3]))/(maxval1*maxval2*maxval3*maxval4)
    matdim4 = matdim4.expand(data.shape[0], data.shape[1], data.shape[2], data.shape[3])


    matdim = matdim1.permute(3, 0, 1, 2) + matdim2.permute(0, 3, 1, 2) + matdim3.permute(0, 1, 3, 2) + matdim4


    return matdim


def ZIdx2Int(inputmat):
    mat = inputmat.view(-1)
    idxpos = torch.tensor(range(mat.numel() ))

    packmat = torch.cat((mat.unsqueeze(-1), idxpos.unsqueeze(-1)), dim=-1)
    packmat = packmat[packmat[:, 0].sort()[1]]
    
    x = torch.cat((packmat, idxpos.unsqueeze(-1)), dim=-1)[:, 1:]
    intZidx = (x[x[:, 0].sort()[1]])[:, 1].squeeze()

    intZidx = intZidx.reshape(inputmat.shape)

    return intZidx



class  Conv2d_FR4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False, groups=1, dilation=1):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
        else:
            self.bias = None



        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_4d(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size))))
        self.register_buffer("IDROP",  torch.zeros(out_channels, in_channels//groups, kernel_size, kernel_size) + 1.0)

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 127)
        
        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1

    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=127):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def setminrate(self, minrate, protectnum=127):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def forward(self, data):


        weight = idct_4d(self.weight)
        output = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)

        return output


class Linear_FR2d(nn.Module):
    def __init__(self, in_features, out_features, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        else:
            self.bias = None

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_2d(torch.empty(out_features, in_features)).squeeze()))
        self.register_buffer("IDROP",   torch.zeros(out_features, in_features).squeeze() + 1.0)

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 255)
        

        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1     


    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=255):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def setminrate(self, minrate, protectnum=255):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), protectnum)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate), 255)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def forward(self, data):


        weight = idct_2d(self.weight)
        output = F.linear(data, weight, self.bias)


        return output



class Conv2d(Conv2d_FR4d):
    pass

class Linear(Linear_FR2d):
    pass



def countParams(model):
    sumnum = 0
    maxnum = -1
    minnum = 10000000000000000
    rate = 0
    totalnum = 0

    for name, layer in model.named_modules():
        try:
            num = layer.IDROP.sum()
            rate = num/layer.IDROP.numel()
            totalnum = totalnum + layer.IDROP.numel()

            if maxnum < num:
                maxnum = num
            
            if minnum > num:
                minnum = num
            
            sumnum += num
        except:
            pass

    rate = sumnum*1.0/totalnum

    return sumnum, minnum, maxnum, rate




def cleanParams(model):
    for name, layer in model.named_modules():
        try:
            idx = layer.IDROP.abs() < 0.0000000000000000000001
            layer.weight.data[idx] = 0
        except:
            pass

    return None




class  ConvTranspose2d_FR4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False, groups=1, directdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels//groups, kernel_size, kernel_size).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
        else:
            self.bias = None

        self.groups = groups

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_4d(torch.empty(in_channels, out_channels//groups, kernel_size, kernel_size))))
        self.register_buffer("IMAT",    torch.zeros(in_channels, out_channels//groups, kernel_size, kernel_size) + 1.0)
        self.register_buffer("IDROP",  torch.zeros(in_channels, out_channels//groups, kernel_size, kernel_size) + 1.0)


        self.minrate = minrate
        self.minnum = max(round(out_channels*in_channels*kernel_size*kernel_size*minrate//groups), 31)
        
        
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()


        self.weightrate = 0
        self.weightnum = -1



        if directdrop:
            self.dropcnt = self.minnum + 1

    def reset(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 16)
        self.IDROP.fill_(1.0)
        self.dropcnt = self.minnum + 10


    def forward(self, data):

        weight = idct_4d(self.weight)
        x = F.conv_transpose2d(data, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
            
        return x


class ConvTranspose2d(ConvTranspose2d_FR4d):
    pass
