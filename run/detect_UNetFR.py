import os
import torch
import torch.nn.functional as F

from unet import UNet
from dice_score import multiclass_dice_coeff, dice_coeff

import numpy as np
import matplotlib.pyplot as plt
import imageio



def loadFiles_plus(path_im, keyword = ""):
    re_fs = []
    re_fullfs = []

    files = os.listdir(path_im)
    files = sorted(files)

    for file in files:
        if file.find(keyword) != -1:
            re_fs.append(file)
            re_fullfs.append(path_im + "/" + file)

    return re_fs, re_fullfs


def model2File(model, save_path):

    state_dict = model.state_dict()
    save_dict = {}

    for i in state_dict.keys():
        save_dict[i] = state_dict[i].detach().to_sparse()

    torch.save(save_dict, save_path)


def file2Model(model, save_path):

    save_dict = torch.load(save_path)
    state_dict = {}

    for i in save_dict.keys():
        state_dict[i] = save_dict[i].to_dense()
    model.load_state_dict(state_dict)

    return model


def countParams(model):
    totalnum = 0
    totalbias = 0
    for name, layer in model.named_modules():
        try:
            num = torch.sum(layer.weight.data.abs() > 0).item()
            totalnum += num  
        except:
            pass

        try:
            num = torch.sum(layer.bias.data.abs() > 0).item()
            totalbias += num
        except:
            pass

    return totalnum, totalbias


def cleanUNet(model):
    for name, layer in model.named_modules():
        try:
            idx = layer.IDROP.abs() < 0.0000000000000000000001
            layer.weight.data[idx] = 0

            layer.IDROP = None
        except:
            pass

        try:
            layer.ZMAT = None
        except:
            pass

        try:
            layer.IMAT = None
        except:
            pass


    return None




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.system("tar -xf unet_fr.tar.xz")
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    cleanUNet(net)

    net = file2Model(net, './unet_fr.pt')
    net.to(device=device)

    numweight, numbias = countParams(net)

    print("number of nonzero parameters in weight:", numweight)
    print("number of nonzero parameters in bias  :", numbias)




    fs_im, fullfs_im = loadFiles_plus('./testimgs/input/', 'png')
    fs_gt, fullfs_gt = loadFiles_plus("./testimgs/mask/", 'png')

    dice_score = 0

    plt.figure(figsize=(12, 4))
    for i in range(len(fullfs_im)):
        
        img = torch.tensor(imageio.imread(fullfs_im[i]), dtype=torch.float32)/255.0
        lab = torch.tensor(imageio.imread(fullfs_gt[i]), dtype=torch.float32)/255.0

        print("segmenting files:", fullfs_im[i])


        img = img.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        lab = lab.unsqueeze(0).to(device=device, dtype=torch.long)

        mask_pred = net(img)


        showmask = mask_pred.argmax(dim=1).squeeze()
        showgt = lab.squeeze()

        lab = F.one_hot(lab, net.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], lab[:, 1:], reduce_batch_first=False)


        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        plt.title("img")

        plt.subplot(1, 3, 2)
        plt.imshow(showgt.detach().cpu().numpy())
        plt.title("Groundtruth")

        plt.subplot(1, 3, 3)
        plt.imshow(showmask.detach().cpu().numpy())
        plt.title("Binary Mask")

        plt.pause(0.1)


    print("average Dice Score:", dice_score.item()/len(fullfs_im))



   

