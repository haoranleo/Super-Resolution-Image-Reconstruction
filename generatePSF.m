function PSF = generatePSF(v, h, I)
% GENERATEPSF - generate a point spread function matrix
% This function generates a PSF (point spread function) matrix to use in a
% matrix multiplication (i.e. not a convolution) with the image I
% V is the vector containing the first non-zero values of the first column
% of the resulting Toeplitz matrix
% H is the vector containing the first non-zero values of the first line
% of the resulting Toeplitz matrix
% I is the image the PSF matrix is for (the resulting PSF matrix will thus
% have the same dimsensions as I)

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

PSF = toeplitz([v zeros(1, size(I, 1)-length(v))], [h zeros(1, size(I, 2)-length(h))]);
PSF = PSF / (sum(v) + sum(h(2:end))); % normalization factor