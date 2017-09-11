function [rot_angle, c] = estimate_rotation(a,dist_bounds,precision)
% ESTIMATE_ROTATION - rotation estimation using algorithm by Vandewalle et al.
%                  用Vandewalle算法估计旋转
%    [rot_angle, c] = estimate_rotation(a,dist_bounds,precision)
%    DIST_BOUNDS gives the minimum and maximum radius to be used
%    DIST_BOUNDS提供了使用到的最小和最大半径（范围）
%    PRECISION gives the precision with which the rotation angle is computed
%    PRECISION提供了计算旋转角度的精度
%    input images A are specified as A{1}, A{2}, etc.

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

h = waitbar(0, 'Rotation Estimation');  %进度条函数
set(h, 'Name', 'Please wait...');   

nr = length(a); % number of inputs 输入的数量
d = 1*pi/180; % width of the angle over which the average frequency value is computed 角宽度，在此基础上进行平均频率值计算
s = size(a{1})/2;
center = [floor(s(1))+1 floor(s(2))+1]; % center of the image and the frequency domain matrix
                                        % 图像中点 ， 频率域矩阵
x = ones(s(1)*2,1)*[-1:1/s(2):1-1/s(2)]; % X coordinates of the pixels  像素的X坐标
y = [-1:1/s(1):1-1/s(1)]'*ones(1,s(2)*2); % Y coordinates of the pixels  像素的Y坐标
x = x(:);
y = y(:);
[th,ra] = cart2pol(x,y); % polar coordinates of the pixels  像素的极坐标

%***********************************************************
DB = (ra>dist_bounds(1))&(ra<dist_bounds(2));
%***********************************************************
th(~DB) = 1000000;
[T, ix] = sort(th); % sort the coordinates by angle theta      Θ的坐标

st = length(T);

%% compute the average value of the fourier transform for each segment
%  计算每一部分傅里叶变换的平均值
I = -pi:pi*precision/180:pi;
J = round(I/(pi*precision/180))+180/precision+1;  %round函数：四舍五入取整
for k = 1:nr
    waitbar(k/(2*nr), h, 'Rotation Estimation');  %进度条
    A{k} = fftshift(abs(fft2(a{k}))); % Fourier transform of the image  
                                      %fft2 2维离散傅里叶快速变换
                                      %fftshift搭配使用，使得fft得出的数据与频率对应
                                     
    ilow = 1;
    ihigh = 1;
    ik = 1;
    for i = 1:length(I)
        ik = ilow;
        while(I(i)-d > T(ik))
            ik = ik + 1;
        end;

        ilow = ik;
        ik = max(ik, ihigh);
        while(T(ik) < I(i)+d)
            ik = ik + 1;
            if (ik > st | T(ik) > 1000)
                break;
            end;
        end;
        ihigh = ik;
        if ihigh-1 > ilow
            h_A{k}(J(i)) = mean(A{k}(ix(ilow:ihigh-1)));
        else
            h_A{k}(J(i)) = 0;
        end
    end;
    v = h_A{k}(:) == NaN;
    h_A{k}(v) = 0;
end

% compute the correlation between h_A{1} and h_A{2-4} and set the estimated rotation angle 
% to the maximum found between -30 and 30 degrees
% 计算h_A{1}和h_A{2-4}的关联并设置估计旋转角为-30度到30度内的最大值对应的度数

H_A = fft(h_A{1});
rot_angle(1) = 0;
c{1} = [];
for k = 2:nr
  H_Binv = fft(h_A{k}(end:-1:1));
  H_C = H_A.*H_Binv;
  h_C = real(ifft(H_C));
  [m,ind] = max(h_C(150/precision+1:end-150/precision));
  rot_angle(k) = (ind-30/precision-1)*precision;
  c{k} = h_C;
end

close(h);
