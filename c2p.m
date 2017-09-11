function impolar = c2p(im)
% IMPOLAR - compute the polar coordinates of the pixels of an image
% IMPOLAR(C2P)-计算图像的像素点的极坐标
%    impolar = c2p(im)
%    convert an image in cartesian coordinates IM
%    to an image in polar coordinates IMPOLAR
%    将笛卡尔坐标系中的图像转换到极坐标系

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

[nrows, ncols] = size(im);

% create the regular rho,theta grid
% 创建常规ρ，Θ网格

r = ones(nrows,1)*[0:nrows-1]/2;
th = [0:nrows-1]'*ones(nrows,1)'*2*pi/nrows-pi;  %'是转置

% convert the polar coordinates to cartesian 
% 将极坐标转化为笛卡尔坐标 
[xx,yy] = pol2cart(th,r); %pol2cart函数用于将极坐标转化为笛卡尔坐标
xx = xx + nrows/2+0.5;
yy = yy + nrows/2+0.5;

% interpolate using bilinear interpolation to produce the final image
% 使用双线性插值生成最终的图像
partx = xx-floor(xx); partx = partx(:);
party = yy-floor(yy); party = party(:);
% floor函数向下取整  (:)是将矩阵转化为向量 按顺序排成一列

impolar = (1-partx).*(1-party).*reshape(im(floor(yy)+nrows*(floor(xx)-1)),[nrows*ncols 1])...
    + partx.*(1-party).*reshape(im(floor(yy)+nrows*(ceil(xx)-1)),[nrows*ncols 1])...
    + (1-partx).*party.*reshape(im(ceil(yy)+nrows*(floor(xx)-1)),[nrows*ncols 1])...
    + partx.*party.*reshape(im(ceil(yy)+nrows*(ceil(xx)-1)),[nrows*ncols 1]);
%reshape函数用于返回与A具有相同元素的N维数组（返回的数组的元素和A中元素相等）

impolar = reshape(impolar,[nrows ncols]);
