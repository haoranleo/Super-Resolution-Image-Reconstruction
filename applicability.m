function [Z X Y] = applicability(a, b, rmax)
% APPLICABILITY：用于产生一个用于规范化卷积的适用性函数

% Z is the applicability matrix and X, Y are the grid coordinates if a 3D
% plot is required.
% Z是应用矩阵，在3D图内X,Y是网络坐标

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

[X Y] = meshgrid(-rmax:rmax, -rmax:rmax);  %meshgrid:网格采样点函数 用于生成3D图

Z = sqrt(X.^2+Y.^2).^(-a).*cos((pi*sqrt(X.^2+Y.^2))/(2*rmax)).^b;
Z = Z .* double(sqrt(X.^2+Y.^2) < rmax); % We want Z=0 outside of rmax   我们想要在rmax范围外Z=0
Z(rmax+1, rmax+1) = 1;