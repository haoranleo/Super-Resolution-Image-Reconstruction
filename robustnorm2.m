function c = robustnorm2(f, f_hat, sigma_noise)
% ROBUSTNORM2 - function used in normalized convolution
% ROBUSTNORM2 - 用于规范化卷积的函数
% sigma_noise is the standard deviation of the input noise
% σ噪声输入噪声的标准差

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%
c = exp(-((f-f_hat).^2) ./ 2*sigma_noise^2);