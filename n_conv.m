function [rec, F] = n_conv(s,delta_est,phi_est,factor, noiseCorrect, TwoPass)
% N_CONV - reconstruct a high resolution image using normalized convolution
%          用标准化卷积重建高分辨率图像
%    [rec, F] = n_conv(s,delta_est,phi_est,factor, noiseCorrect, TwoPass)
%    reconstruct an image with FACTOR times more pixels in both dimensions
%    using normalized convolution on the pixels from the images in S
%    (S{1},...) and using the shift and rotation information from DELTA_EST 
%    and PHI_EST; options are available to specify whether a noise correction
%    step and a second pass should be applied or not
%    利用图像S像素的标准化卷积和DELTA_EST和PHI_EST的旋转、位移信息重建在两个方向（横轴和纵轴）具有更多像素点的FACTOR矩阵

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

if nargin < 4
    errordlg('Not enough input arguments', 'Error...');
elseif nargin < 5
    noiseCorrect = false;
    TwoPass = false;
end

n=length(s);
ss = size(s{1});
if (length(ss)==3) 
    error('This function only takes 2-dimensional matrices'); 
end
center = (ss+1)/2;
phi_rad = phi_est*pi/180;

% compute the coordinates of the pixels from the N images, using DELTA_EST and PHI_EST
% 利用DELTA_EST和PHI_EST计算N图像像素的坐标
for i=1:n % for each image 对每个图像
    s_c{i}=s{i};
    s_c{i} = s_c{i}(:);
    r{i} = [1:factor:factor*ss(1)]'*ones(1,ss(2)); % create matrix with row indices 创建矩阵的行索引（指数）
    c{i} = ones(ss(1),1)*[1:factor:factor*ss(2)]; % create matrix with column indices  列
    r{i} = r{i}-factor*center(1); % shift rows to center around 0   将行移到0附近的中心
    c{i} = c{i}-factor*center(2); % shift columns to center around 0
    coord{i} = [c{i}(:) r{i}(:)]*[cos(phi_rad(i)) sin(phi_rad(i)); -sin(phi_rad(i)) cos(phi_rad(i))]; % rotate 
    r{i} = coord{i}(:,2)+factor*center(1)+factor*delta_est(i,1); % shift rows back and shift by delta_est
                                                                 % 用delta_est将行向回移                                                              
    c{i} = coord{i}(:,1)+factor*center(2)+factor*delta_est(i,2); % shift columns back and shift by delta_est
    rn{i} = r{i}((r{i}>0)&(r{i}<=factor*ss(1))&(c{i}>0)&(c{i}<=factor*ss(2)));
    cn{i} = c{i}((r{i}>0)&(r{i}<=factor*ss(1))&(c{i}>0)&(c{i}<=factor*ss(2)));
    sn{i} = s_c{i}((r{i}>0)&(r{i}<=factor*ss(1))&(c{i}>0)&(c{i}<=factor*ss(2)));
end

s_ = []; r_ = []; c_ = []; sr_ = []; rr_ = []; cr_ = [];
for i=1:n % for each image 
    s_ = [s_; sn{i}];
    r_ = [r_; rn{i}];
    c_ = [c_; cn{i}];
end
clear s_c r c coord rn cn sn

% Apply the normalized convolution algorithm  应用标准化卷积算法
if nargout == 2
    [rec, F] = n_convolution(c_,r_,s_,ss*factor,factor, s{1}, noiseCorrect, TwoPass);
else
    rec = n_convolution(c_,r_,s_,ss*factor,factor, s{1}, noiseCorrect, TwoPass);
end

rec(isnan(rec))=0;   %insan用于判断数组中的数字是否为无穷大
