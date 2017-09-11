function [delta_est, phi_est] = estimate_motion(s,r_max,d_max)
% ESTIMATE_MOTION - shift and rotation estimation using algorithm by Vandewalle et al.
%                 -用Vandewalle算法对位移和旋转进行估计

%    [delta_est, phi_est] = estimate_motion(s,r_max,d_max)
%    R_MAX is the maximum radius in the rotation estimation
%    R_MAX是旋转估计的最大半径  

%    D_MAX is the number of low frequency components used for shift estimation
%    D_MAX是用于位移估计的低频成分的数量

%    input images S are specified as S{1}, S{2}, etc.
%    输入的图片A依次被指定为S{1},S{2}等
%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

if (nargin==1) % default values  无效值
   r_max = 0.8;
   d_max = 8;
end

% rotation estimation
% 旋转估计
[phi_est, c_est] = estimate_rotation(s,[0.1 r_max],0.1);

% rotation compensation, required to estimate shifts
% 旋转所得的结果，也用于位移估计
s2{1} = s{1};
nr=length(s);
for i=2:nr
    s2{i} = imrotate(s{i},-phi_est(i),'bicubic','crop');
end

% shift estimation
% 位移估计
delta_est = estimate_shift(s2,d_max);


