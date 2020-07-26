import torch
import torch.nn as nn
import math
import numpy as np
import os
from os import listdir
from os.path import join
import torchvision.transforms as transforms
import torch.nn.functional as F
from math import log10
from skimage import measure

def save_checkpoint(model, epoch, model_folder):
  model_out_path = "checkpoints/%s/%d.pth" % (model_folder, epoch)

  state_dict = model.module.state_dict()
  for key in state_dict.keys():
    state_dict[key] = state_dict[key].cpu()

  if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

  if not os.path.exists("checkpoints/" + model_folder):
    os.makedirs("checkpoints/" + model_folder)

  torch.save({
    'epoch': epoch,
    'state_dict': state_dict}, model_out_path)
  print("Checkpoint saved to {}".format(model_out_path))

def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]
    return ssim_list

def edge_compute(x):
    x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,:,1:] += x_diffx
    y[:,:,:,:-1] += x_diffx
    y[:,:,1:,:] += x_diffy
    y[:,:,:-1,:] += x_diffy
    y = torch.sum(y,1,keepdim=True)/3
    y /= 4
    return y