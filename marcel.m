function [delta_est, phi_est] = marcel(s,M)
% MARCEL - estimate shift and rotation parameters using Marcel et al. algorithm
%          用Marcel算法进行位移和旋转参数的估计
%    [delta_est, phi_est] = marcel(s,M)
%    horizontal and vertical shifts DELTA_EST and rotations PHI_EST are 
%    estimated from the input images S (S{1},etc.). For the shift and 
%    rotation estimation, the Fourier transform images are interpolated by 
%    a factor M to increase precision
%    根据输入的图像矩阵S进行水平方向和竖直方向位移DELTA_EST矩阵以及旋转PHI_EST矩阵的估计。
%    为提高旋转和位移的估计精度，将对傅里叶变换后的图像用参数M进行插值

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%
nr=length(s);
S = size(s{1});
if (nargin==1)
    M = 10; % magnification factor to have higher precision
            % 设置放大系数以提高精度
end

% if the image is not square, make it square (rest is not useful for rotation estimation anyway)
% 如果图片不是正方形的，则使其变为正方形（因为其余部分对于旋转估计没有用处）
if S(1)~=S(2)
    if S(1)>S(2)
        for i=1:length(s)
           s{i} = s{i}(floor((S(1)-S(2))/2)+1:floor((S(1)-S(2))/2)+S(2),:);
        end
    else
        for i=1:length(s)
           s{i} = s{i}(:,floor((S(2)-S(1))/2)+1:floor((S(2)-S(1))/2)+S(1));
        end
    end
end

phi_est = zeros(1,nr);
r_ref = S(1)/2/pi;
IMREF = fft2(s{1});
IMREF_C = abs(fftshift(IMREF));
IMREF_P = c2p(IMREF_C);
IMREF_P = IMREF_P(:,round(0.1*r_ref):round(1.1*r_ref)); % select only points with radius 0.1r_ref<r<1.1r_ref
                                                        % 选择在环形内的点（0.1~1.1)
IMREF_P_ = fft2(IMREF_P);
for i=2:nr
    % rotation estimation  旋转估计
    IM = abs(fftshift(fft2(s{i})));
    IM_P = c2p(IM);
    IM_P = IM_P(:,round(0.1*r_ref):round(1.1*r_ref)); % select only points with radius 0.1r_ref<r<1.1r_ref
    IM_P_ = fft2(IM_P);
    psi = IM_P_./IMREF_P_;
    PSI = fft2(psi,M*S(1),M*S(2));
    [m,ind] = max(PSI);
    [mm,iind] = max(m);
    phi_est(i) = (ind(iind)-1)*360/S(1)/M;

    % rotation compensation, required to estimate shifts  根据位移的估计补偿旋转参数
    s2{i} = imrotate(s{i},-phi_est(i),'bilinear','crop');

    % shift estimation   位移估计
    IM = fft2(s2{i});
    psi = IM./IMREF;
    PSI = fft2(psi,M*S(1),M*S(2));
    [m,ind] = max(PSI);
    [mm,iind] = max(m);
    delta_est(i,1) = (ind(iind)-1)/M;
    delta_est(i,2) = (iind-1)/M;
    if delta_est(i,1)>S(1)/2
        delta_est(i,1) = delta_est(i,1)-S(1);
    end
    if delta_est(i,2)>S(2)/2
        delta_est(i,2) = delta_est(i,2)-S(2);
    end
end

% Sign change in order to follow the project standards  
% 为了按照项目的标准采用单变换
delta_est = -delta_est;
