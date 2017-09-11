function delta_est = estimate_shift(s,n)
% ESTIMATE_SHIFT - shift estimation using algorithm by Vandewalle et al.
%                - 用Vandewalle算法估计位移
%    delta_est = estimate_shift(s,n)
%    estimate shift between every image and the first (reference) image
%    对每一幅图像和第一幅图像估计位移
%    N specifies the number of low frequency pixels to be used
%    参数N指定了用到的低频像素点的数量
%    input images S are specified as S{1}, S{2}, etc.

%    DELTA_EST is an M-by-2 matrix with M the number of images
%    DELTA_EST 是一个M*2的矩阵，M是图像的数量

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

h = waitbar(0, 'Shift Estimation');    %进度条
set(h, 'Name', 'Please wait...');

nr = length(s);
delta_est=zeros(nr,2);
p = [n n]; % only the central (aliasing-free) part of NxN pixels is used for shift estimation
           % N*N像素点中只有中间发生混叠的部分才被用作位移估计
sz = size(s{1});
S1 = fftshift(fft2(s{1})); % Fourier transform of the reference image
                           %对参考图像进行2维快速傅里叶变换
for i=2:nr
  waitbar(i/nr, h, 'Shift Estimation');
  S2 = fftshift(fft2(s{i})); % Fourier transform of the image to be registered
                             %对将要进行配准的图像进行傅里叶变换
                               
  S2(S2==0)=1e-10;
  Q = S1./S2;
  A = angle(Q); % phase difference between the two images
                %两幅图像的相位差
                
  % determine the central part of the frequency spectrum to be used
  %确定要使用的频率谱的中心部分
  beginy = floor(sz(1)/2)-p(1)+1;
  endy = floor(sz(1)/2)+p(1)+1;
  beginx = floor(sz(2)/2)-p(2)+1;
  endx = floor(sz(2)/2)+p(2)+1;
  
  % compute x and y coordinates of the pixels
  %计算像素点的x和y坐标
  x = ones(endy-beginy+1,1)*[beginx:endx];
  x = x(:);
  y = [beginy:endy]'*ones(1,endx-beginx+1);
  y = y(:);
  v = A(beginy:endy,beginx:endx);
  v = v(:);

  % compute the least squares solution for the slopes of the phase difference plane
  % 计算相位差平面斜率的最小二乘解
  
  M_A = [x y ones(length(x),1)];
  r = M_A\v;
  delta_est(i,:) = -[r(2) r(1)].*sz/2/pi;
end

close(h);