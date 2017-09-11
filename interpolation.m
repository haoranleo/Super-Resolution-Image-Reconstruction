function rec = interpolation(s,delta_est,phi_est,factor)
% INTERPOLATION - reconstruct a high resolution image using bicubic interpolation
%               - 用双立方插值重建高分辨率图像
%    rec = interpolation(s,delta_est,phi_est,factor)
%    reconstruct an image with FACTOR times more pixels in both dimensions
%    using bicubic interpolation on the pixels from the images in S
%    (S{1},...) and using the shift and rotation information from DELTA_EST 
%    and PHI_EST
%    通过对S储存图像的像素信息进行双立方插值，以及利用DELTA_EST和PHI_EST携带的位移和旋转信息
%    重建更高倍像素的图像FACTOR矩阵

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

n=length(s);
ss = size(s{1});
if (length(ss)==2) ss=[ss 1]; end
center = (ss+1)/2;
phi_rad = phi_est*pi/180;

% compute the coordinates of the pixels from the N images, using DELTA_EST and PHI_EST
%利用DELTA_EST和PHI_EST矩阵计算N图像的像素坐标
for k=1:ss(3) % for each color channel  对每一个彩色信号通道(色度?)
  for i=1:n % for each image 对每一个图像
    s_c{i}=s{i}(:,:,k);
    s_c{i} = s_c{i}(:);
    r{i} = [1:factor:factor*ss(1)]'*ones(1,ss(2)); % 	 
    c{i} = ones(ss(1),1)*[1:factor:factor*ss(2)]; % create matrix with column indices 创建行索引的矩阵
    r{i} = r{i}-factor*center(1); % shift rows to center around 0   
    c{i} = c{i}-factor*center(2); % shift columns to center around 0 
    coord{i} = [c{i}(:) r{i}(:)]*[cos(phi_rad(i)) sin(phi_rad(i)); -sin(phi_rad(i)) cos(phi_rad(i))]; % rotate 
    r{i} = coord{i}(:,2)+factor*center(1)+factor*delta_est(i,1); % shift rows back and shift by delta_est
                                                                 % 利用delta_est往回移动行
    c{i} = coord{i}(:,1)+factor*center(2)+factor*delta_est(i,2); % shift columns back and shift by delta_est
                                                                 % 利用delta_est往回移动列
    rn{i} = r{i}((r{i}>0)&(r{i}<=factor*ss(1))&(c{i}>0)&(c{i}<=factor*ss(2)));
    cn{i} = c{i}((r{i}>0)&(r{i}<=factor*ss(1))&(c{i}>0)&(c{i}<=factor*ss(2)));
    sn{i} = s_c{i}((r{i}>0)&(r{i}<=factor*ss(1))&(c{i}>0)&(c{i}<=factor*ss(2)));
  end

  s_ = []; r_ = []; c_ = []; sr_ = []; rr_ = []; cr_ = [];
  for i=1:n % for each image 对每个图像
    s_ = [s_; sn{i}];
    r_ = [r_; rn{i}];
    c_ = [c_; cn{i}];
  end
  clear s_c r c coord rn cn sn
  
  h = waitbar(0.5, 'Image Reconstruction');
  set(h, 'Name', 'Please wait...');
  
  % interpolate the high resolution pixels using cubic interpolation 
  % 用立方插值对高分辨率的像素点进行插值
  rec_col = griddata(c_,r_,s_,[1:ss(2)*factor],[1:ss(1)*factor]','cubic'); % option QJ added to make it work 
  rec(:,:,k) = reshape(rec_col,ss(1)*factor,ss(2)*factor);
end
rec(isnan(rec))=0;

close(h);