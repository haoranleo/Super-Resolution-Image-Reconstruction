function s_low = lowpass(s,part)
% LOWPASS - low-pass filter an image by setting coefficients to zero in frequency domain
%         - 通过设置频域内系数为零对图像进行低通滤波
%    s_low = lowpass(s,part)
%    S is the input image S为输入图像
%    PART indicates the relative part of the frequency range [0 pi] to be kept
%    PART矩阵表示保存的频率变化域为0~pi的相关部分

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

% set coefficients to zero 设置系数为零
if size(s,1)>1 % 2D signal 两个维度的信号
    S = fft2(s); % compute the Fourier transform of the image 计算图像的傅里叶变换
    S(round(part(1)*size(S,1))+1:round((1-part(1))*size(S,1))+1,:) = 0;
    S(:,round(part(2)*size(S,2))+1:round((1-part(2))*size(S,2))+1) = 0;
    s_low = real(ifft2(S)); % compute the inverse Fourier transform of the filtered image   
                            % 计算滤波后图像的傅里叶反变换
else % 1D signal 一个维度的信号
    S = fft(s); % compute the Fourier transform of the image  计算图像的傅里叶变换
    S(round(part*length(S))+1:round((1-part)*length(S))+1) = 0;
    s_low = real(ifft(S)); % compute the inverse Fourier transform of the filtered signal  计算滤波后信号的傅里叶反变换
end
