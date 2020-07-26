import sys
import numpy as np
import cv2
import math
from net import *
import torchvision.transforms as transforms
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from PIL import Image
import os

#寻找暗通道
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

#估计大气光值
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/100),1))
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)
    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A

#导向滤波
def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
    q = mean_a*im + mean_b
    return q

#使用导向滤波对估计的投射图精细化
def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 50
    eps = 0.001
    t = Guidedfilter(gray,et,r,eps)
    return t

#根据投射图与全球大气光恢复出无雾图像
def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res

def load_model(model_path,gpu = 1):
    print("==========> Building model")
    #设定要验证的模型
    model = DehazeNet()
    #设定模型对应的训练好的pth文件
    checkpoint = torch.load(model_path)
    #加载权重
    model.load_state_dict(checkpoint["state_dict"])
    #将模型放在GPU中
    if gpu:
        model = nn.DataParallel(model, device_ids=[i for i in range(1)]).cuda()
    return model

if __name__ == '__main__':
    cudnn.benchmark = True
    #读取模型
    model = load_model('./best.pth')
    #读取模型之后，进行前向传播
    #===== Load input image =====
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]
    )
    model.eval()

    file_dir = "../data/test/SOTS/indoor/hazy/"
    #读取文件夹中的所有图像
    for root, dirs, files in os.walk(file_dir):  
        for i in range(len(files)):
            #print(files) #当前路径下所有非目录子文件 
            imagePath = file_dir+'/'+files[i]

            src = Image.open(imagePath)
            src = np.array(src)
            #对三通道图像进行填充
            npad = ((7,8), (7,8), (0,0))
            im = np.pad(src, npad, 'symmetric')
            #转化为tensor
            imgIn = transform(im).unsqueeze_(0)
            #===== Test procedures =====
            varIn = Variable(imgIn)
            with torch.no_grad():
                output = model(varIn)
            te = output.data.cpu().numpy().squeeze()
            save_path = "./trans"
            if  not os.path.exists(save_path):#如果路径不存在
                os.makedirs(save_path)
            cv2.imwrite(save_path+"/"+files[i],te*255)