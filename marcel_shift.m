function delta_est = marcel_shift(s,M)
% MARCEL_SHIFT shift estimation using algorithm by Marcel et al.
%              用Marcel算法估计位移
%    [delta_est, phi_est] = marcel(s,M)
%    motion estimation algorithm implemented from the paper by Marcel et al.
%    horizontal and vertical shifts DELTA_EST are estimated from the input 
%    images S (S{1},etc.). For the shift estimation, the Fourier transform 
%    images are interpolated by a factor M to increase precision.
%    根据输入的图像矩阵S估计水平和竖直方向的位移DELTA_EST
%    在位移估计方面，将对傅里叶变换后的图像用参数M进行插值以提高精度。

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%                    

nr=length(s);
S = size(s{1});
if (nargin==1)
    M = 10; % magnification factor to have higher precision
            % 为更高的精度设定放大系数
end

phi_est = zeros(1,nr);
r_ref = S(1)/2/pi;
IMREF = fft2(s{1});
IMREF_C = abs(fftshift(IMREF));
IMREF_P = c2p(IMREF_C);
IMREF_P = IMREF_P(:,round(0.1*r_ref):round(1.1*r_ref)); % select only points with radius 0.1r_ref<r<1.1r_ref
                                                        % 选择在内外径分别为0.1r_ref和1.1r_ref的环形区域内的点
IMREF_P_ = fft2(IMREF_P);
for i=2:nr
    % shift estimation    位移估计
    IM = fft2(s{i});
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
