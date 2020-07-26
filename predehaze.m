clc;
clear;
close all;
hazydir = './test_outdoor/'
transdir = './trans_outdoor/'
savedir = './results_outdoor/'
imgset=dir(hazydir);
imgset=imgset(3:end);
nimg=size(imgset,1);
for k=1:nimg
    img_name = imgset(k).name;  
    hazy_name=strcat(hazydir,img_name);
    trans_name = strcat(transdir,img_name);

    haze=imread(hazy_name);
    im = double(haze)./255;
    r0 = 50;
    eps = 10^-3; 
    gray_I = rgb2gray(im);
    F4 = imread(trans_name);
    F4 = double(F4)./255;

    %% Atmospheric light
    %降序排序
    sortdata = sort(F4(:), 'ascend');
    %取前百分之一
    idx = round(0.01 * length(sortdata));
    val = sortdata(idx); 
    id_set = find(F4 <= val);
    BrightPxls = gray_I(id_set);
    iBright = BrightPxls >= max(BrightPxls);
    id = id_set(iBright);
    Itemp=reshape(im,size(im,1)*size(im,2),size(im,3));
    A = mean(Itemp(id, :),1);
    A=reshape(A,1,1,3);

    F4 = guidedfilter(gray_I, F4, r0, eps);

    J=bsxfun(@minus,im,A);
    J=bsxfun(@rdivide,J,F4);
    J=bsxfun(@plus,J,A);
    dehaze=J;
    savename = strcat(savedir,img_name);
    imwrite(dehaze,savename);
end
