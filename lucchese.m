function [delta_est, phi_est] = lucchese(s,M)
% LUCCHESE - estimate shift and rotation parameters using Lucchese and Cortelazzo algorithm
%          - 用Lucchese和Cortelazzo算法估计位移和旋转参数
%    [delta_est, phi_est] = lucchese(s,M)
%    horizontal and vertical shifts DELTA_EST and rotations PHI_EST are 
%    estimated from the input images S (S{1},etc.). 
%    根据输入的S图像估计水平和竖直方向的位移DELTA_EST以及旋转角PHI_EST
%    For the shift estimation, the Fourier transform images are interpolated by a factor 
%    M to increase precision
%    对于位移估计，图像的傅里叶变换由参数M进行插值以提高精度

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%
nr=length(s);
S = size(s{1});
if (nargin==1)
   M = 4; % magnification factor to increase precision of shift estimation
          % 用于提升位移估计精度的放大系数     
end    

% ROTATION ESTIMATION  旋转估计

z = zeros(2*S);
for i=1:nr
    s2{i} = imresize(s{i},0.5,'bicubic');
end

% init stage 1 - coarse estimation of phi   phi粗略估计
Rmax = 80;
interpolation = 4;
histsize = 2;
IMREF = fftshift(fft2(s2{1},2*S(1),2*S(2)));
IMREF_SHIFT = fft2(s{1});
for i=2:nr
    % stage 1 - coarse estimation of phi
    s2{i}=s2{i}(:,end:-1:1);
    IM = fftshift(fft2(s2{i},2*S(1),2*S(2)));
    Delta = abs(IMREF.^2)/abs(IMREF(S(1)+1,S(2)+1)^2) ...
        - abs(IM.^2)/abs(IM(S(1)+1,S(2)+1)^2); % S is even  偶的
    Delta_part = Delta(S(1)-Rmax:S(1)+Rmax,S(2)-Rmax:S(2)+Rmax);
    Delta_part = imresize(Delta_part,interpolation,'bilinear');
    ds = size(Delta_part);
    locus = contourc(Delta_part,[0 0]);
    ind = find(locus(1,:)>0); % exclude the 'index' columns with 0 and the length of the contour
                              % 除去第0列'index'和轮廓的长度
    locus = locus(:,ind)-[326.5;326.5]*ones(1,length(ind));
    [th,r] = cart2pol(locus(1,:),locus(2,:));
    th2a = th(find(r<Rmax & th>-pi/4 & th<pi/4)); % only keep points for which r<Rmax 只保留r<RMAX的点
    th2b = th(find(r<Rmax & th>pi/4 & th<3*pi/4)); % only keep points for which r<Rmax
    H1 = hist(th2a*180/pi,[-45:1/2/histsize:45]);
    H2 = hist(th2b*180/pi,[45:1/2/histsize:135]);
    H = H1+H2;
    H = filter(gausswin(11),1,H);
    H = H(6:end); % need to cut off negative part of gaussian from filtering 需要去掉高斯滤波的负部分
                  % gaussian was not centered 高斯滤波没有中心
    [m,ind] = max(H);
    phi_est1(i) = (ind-histsize*90-1)/histsize;
    %figure; plot(H);

    % stage 2 - refinement of stage 1
    gamma = 4.5; % 1/2 aperture in degrees  0.5孔度
    histsize_ = 100;
    th2a_ = th(find(r<Rmax & th>(phi_est1(i)/2-gamma)*pi/180 ...
        & th<(phi_est1(i)/2+gamma)*pi/180)); % only keep points for which r<Rmax 只保留r<RMAX的点
    th2b_ = th(find(r<Rmax & th>(phi_est1(i)/2-gamma)*pi/180+pi/2 ...
        & th<(phi_est1(i)/2+gamma)*pi/180+pi/2)); % only keep points for which r<Rmax
    H1_ = hist(th2a_*180/pi,[-gamma:1/2/histsize_:gamma]);
    H2_ = hist(th2b_*180/pi,[-gamma:1/2/histsize_:gamma]);
    H_ = H1_+H2_;
    H_ = filter(gausswin(11),1,H_);  %gausswin(N)用于返回在列向量内的N点高斯窗，用于绘制高斯窗。
    Hnorm = H_/sum(H_);
    theta = [-gamma:1/2/histsize_:gamma];
    phi_est2(i) = phi_est1(i)+sum(theta.*Hnorm);
    sigma2 = sum(theta.*theta.*Hnorm);
    
    % stage 3
    omega = find(r<Rmax & ...
        ((th>(phi_est2(i)/2-sigma2)*pi/180 & th<(phi_est2(i)/2+sigma2)*pi/180) ...
        | (th>(phi_est2(i)/2-sigma2)*pi/180 + pi & th<(phi_est2(i)/2+sigma2)*pi/180 + pi) ...
        | (th>(phi_est2(i)/2-sigma2)*pi/180 - pi & th<(phi_est2(i)/2+sigma2)*pi/180 - pi)));
    omega_ = find(r<Rmax & ...
        ((th>(phi_est2(i)/2-sigma2)*pi/180+pi/2 & th<(phi_est1(i)/2+sigma2)*pi/180+pi/2) ...
        | (th>(phi_est2(i)/2-sigma2)*pi/180+3*pi/2 & th<(phi_est1(i)/2+sigma2)*pi/180+3*pi/2) ...
        | (th>(phi_est2(i)/2-sigma2)*pi/180-pi/2 & th<(phi_est1(i)/2+sigma2)*pi/180-pi/2)));
    kx = locus(1,omega);
    ky = locus(2,omega);
    kx_ = locus(1,omega_);
    ky_ = locus(2,omega_);
    %figure; plot(locus(1,:),locus(2,:),'.'); hold; plot(kx,ky,'ro'); plot(kx_,ky_,'rx');title('s3')
    phi_est3(i) = atan(2*(sum(kx.*ky)/length(kx) - sum(kx_.*ky_)/length(kx_))/...
        (sum(kx.*kx-ky.*ky)/length(kx) - sum(kx_.*kx_-ky_.*ky_)/length(kx_)))*180/pi;

    phi_est = phi_est3;
    
    % rotation compensation, required to estimate shifts 根据得到的位移的估计，补偿旋转参数
    s2{i} = imrotate(s{i},-phi_est(i),'bilinear','crop');

    % SHIFT ESTIMATION     位移估计
    IM = fft2(s2{i});
    psi = IM./IMREF_SHIFT;
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
