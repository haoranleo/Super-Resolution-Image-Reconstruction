function [s, ground_truth] = create_images(im,delta,phi,scale,nr,snr)
% CREATE_IMAGES - generate low resolution shifted and rotated images
%               - 产生低分辨率的存在位移和旋转的图像

%    [s, ground_truth] = create_images(im,delta,phi,scale,nr,snr)
%    create NR low resolution images from a high resolution image IM,
%     NR为低分辨率图像矩阵 IM为高分辨率图像矩阵

%    with shifts DELTA (multiples of 1/8) and rotation angles PHI (degrees)
%     DELTA为位移矩阵（1/8的倍数） PHI为旋转角度矩阵（度）

%    the low resolution images have a factor SCALE less pixels
%    in both dimensions than the input image IM
%    低分辨率图像有一个factor SCALE ，它的像素在两个维度上都比输入的图像要少

%    if SNR is specified, noise is added to the different images to obtain
%    the given SNR value
%    如果SNR指定，噪声便会加到不同的图像以保持给定的SNR值

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

im=resample(im,2,1)'; % upsample the image by 2
im=resample(im,2,1)'; %resample为采样函数，以2Hz的频率进行采样
for i=2:nr
    im2{i} = shift(im,-delta(i,2)*2*scale,-delta(i,1)*2*scale); % shift the images by integer pixels
                                                                %以整数个像素移动图像
    if (phi(i) ~= 0)
      im2{i} = imrotate(im2{i},phi(i),'bicubic','crop'); % rotate the images
                                                         %如果旋转角度不为零，则旋转图像
    end
end
im2{1} = im; % the first image is not shifted or rotated
             %第一幅低分辨率图像即没有位移也没有发生旋转
for i=1:nr
    im2{i} = lowpass(im2{i},[0.12 0.12]); % low-pass filter the images   对图像进行低通滤波
                                            % such that they satisfy the conditions specified in the paper
                                            %这样它们就满足了要求的条件
                                            % a small aliasing-free part of the frequency domain is needed
                                            %一个小的频率域的（自由？）混叠是必须的
    if (i==1) % construct ground truth image as reconstruction target
               %确定母图像（基础图像）为重建的目标
     ground_truth=downsample(im2{i},4)';  %downsample为取样函数
     ground_truth=downsample(ground_truth,4)';
    end
    im2{i} = downsample(im2{i},2*scale)'; % downsample the images by 8  以8为单位进行取样
    im2{i} = downsample(im2{i},2*scale)';
end

% add noise to the images (if an SNR was specified)
%如果SNR参数被指定，则向图像中加入噪声
if (nargin==6)
  for i=1:nr
    S = size(im2{i});
    n = randn(S(1),S(2));   %randn(m,n)：返回一个m*n的随机项矩阵
    n = sqrt(sum(sum(im2{i}.*im2{i}))/sum(sum(n.*n)))*10^(-snr/20)*n;
    s{i} = im2{i}+n; %加入噪声
    %snr = 10*log10(sum(sum(im2{i}.*im2{i}))/sum(sum(n.*n)))
  end
else
  s=im2;
end
