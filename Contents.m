% SUPERRESOLUTION - Graphical User Interface for Super-Resolution Imaging
% SUPERRESOLUTION - 超分辨率成像的图形用户界面
% This program allows a user to perform registration of a set of low 
% resolution input images and reconstruct a high resolution image from them.
% 这个程序可以使用户从一系列低分辨率的输入图像来重建一个高分辨率的图像
% Multiple image registration and reconstruction methods have been
% implemented. As input, the user can either select existing images, or 
% generate a set of simulated low resolution images from a high resolution 
% image. 
% 多种图像配准和图像重建的方法被应用到。输入方面用户可以选择已经存在的低分辨率图像或者利用
% 高分辨率图像生成一系列模拟低分辨率图像
% More information is available online:
% 更多信息可在线上查询
% http://lcavwww.epfl.ch/software/superresolution
% If you use this software for your research, please also put a reference
% to the related paper 
% 如果你将这个软件用于研究，请注明参考了本软件
% "A Frequency Domain Approach to Registration of Aliased Images            
% with Application to Super-Resolution" 
% 频域方法
% Patrick Vandewalle, Sabine Susstrunk and Martin Vetterli                  
% available at http://lcavwww.epfl.ch/reproducible_research/VandewalleSV05/ 

% v 1.0 - January 12, 2006 by Patrick Vandewalle, Patrick Zbinden and Cecilia Perez
% v 2.0 - November 6, 2006 by Patrick Vandewalle and Karim Krichane

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

% Graphical User Interfaces
%   superresolution   - main program
%                     - 主程序
%   generation        - generate a set of low resolution shifted and 
%                       rotated images from a single input image
%                     -从输入的一幅图像产生一系列低分辨率的存在旋转和位移的图像
%
% Image Registration  图像配准模块
%   estimate_motion   - shift and rotation estimation using algorithm 
%                       by Vandewalle et al.
%                     -用Vandewalle算法进行图像位移和旋转的估计

%   estimate_rotation - rotation estimation using algorithm by Vandewalle et al.
%                     -用Vandewalle算法进行图像旋转的估计

%   estimate_shift    - shift estimation using algorithm by Vandewalle et al.
%                     -用Vandewalle算法进行图像位移的估计

%   keren             - estimate shift and rotation parameters 
%                       using Keren et al. algorithm
%                     -用Keren算法进行位移和旋转参数的估计

%   keren_shift       - estimate shift parameters using Keren et al. algorithm
%                     -用Keren算法进行位移参数的估计

%   lucchese          - estimate shift and rotation parameters 
%                       using Lucchese and Cortelazzo algorithm
%                     -用Lucchese和Cortelazzo算法进行位移和旋转的估计

%   marcel            - estimate shift and rotation parameters 
%                       using Marcel et al. algorithm
%                     -用Marcel算法进行位移和旋转参数的估计

%   marcel_shift      - estimate shift parameters using Marcel et al. algorithm
%                     -用Marcel算法进行位移参数的估计
 
% Image Reconstruction  图像重建模块
%   interpolation     - reconstruct a high resolution image from a set of 
%                       low resolution images and their registration parameters
%                       using bicubic interpolation
%                     -使用双立方插值对一系列低分辨率图像以及他们配准的信息进行重建

%   iteratedbackprojection - reconstruct a high resolution image from a set of 
%                       low resolution images and their registration parameters
%                       using iterated backprojection
%                       -迭代投影法

%   n_conv (and n_convolution) - reconstruct a high resolution image from a set
%                       of low resolution images and their registration parameters
%                       using algorithm by Pham et al.
%                       -Pham算法

%   papoulisgerchberg - reconstruct a high resolution image from a set of 
%                       low resolution images and their registration parameters
%                       using algorithm by Papoulis and Gerchberg
%                       -Papoulis和Gerchberg算法

%   pocs              - reconstruct a high resolution image from a set of 
%                       low resolution images and their registration parameters
%                       using POCS (projection onto convex sets) algorithm
%                     - 凸集投影法（POCS)

%   robustSR          - reconstruct a high resolution image from a set of 
%                       low resolution images and their registration parameters
%                       using robust super-resolution algorithm by Zomet et al.
%                     - Zomet的健壮超分辨率算法

% Helper Functions    -辅助模块
%   applicability     - compute the applicability function in normalized 
%                       convolution method
%                     - 产生用于规范化卷积的适用性函数

%   c2p               - compute the polar coordinates of the pixels of an image
%                     - 计算图像像素点的极坐标

%   create_images     - generate low resolution shifted and rotated images
%                       from a single high resolution input image
%                     - 利用输入的一副高分辨率图像产生一系列低分辨率的存在位移和旋转的图像

%   generate_PSF      - generate the point spread function (PSF) matrix
%                     - 产生 点扩散函数矩阵（PSF)

%   lowpass           - low-pass filter an image by setting coefficients 
%                       to zero in frequency domain
%                     - 通过设置频域内系数为零对图像进行低通滤波

%   robustnorm2       - function used in normalized convolution reconstruction
%                     - 用于规范化卷积重建的函数

%   shift             - shift an image over a non-integer amount of pixels
%                     - 使图像发生一个非整数像素的位移（亚像素位移）

