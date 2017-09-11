function [delta_est, phi_est] = keren(im)
% KEREN - estimate shift and rotation parameters using Keren et al. algorithm
%       - 利用Keren算法估计位移和旋转参数
%    [delta_est, phi_est] = keren(im)

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%
% 本算法为Keren配准，配准算法能够估计低分辨率图像帧间亚像素级的运动位移参数；
% 考虑到实时应用的要求，该配准算法具有计算负载低，执行速度有效的特点；
% Keren算法利用一阶Taylor展开的空域配准算法，精度较高，对平移量和旋转量的估计精准；
% 是超分辨率重建中广泛使用的一种亚像素级配准方式，具有较强的鲁棒性；
%% -----------------------------------------------------------
for imnr = 2:length(im)
    % construct pyramid scheme
    % 构建金字塔体
    lp = fspecial('ga',5,1);     % fspecial 用于建立预定义的滤波算子
    % 此处为高斯低通滤波 有两个参数 hsize:5 表示模板尺寸， sigma:1 表示滤波器的标准值
    im0{1} = im{1};
    im1{1} = im{imnr};
    for i=2:3
        im0{i} = imresize(conv2(im0{i-1},lp,'same'),0.5,'bicubic');
        im1{i} = imresize(conv2(im1{i-1},lp,'same'),0.5,'bicubic');
        % imresize 改变图像大小，B = imresize(A,[mrows ncols],method)
        % 使用由method指定的插值运算来改变图像的大小，此处选用bicubic，即双三次插值算法
        % conv2 二维矩阵卷积运算，same表示返回与第一项大小相同卷积值的中间部分
    end
    
    stot = zeros(1,3);
    % do actual registration, based on pyramid
    % 根据金字塔阵配准图像
    for pyrlevel=3:-1:1
        f0 = im0{pyrlevel};
        f1 = im1{pyrlevel};
        
        [y0,x0]=size(f0);
        xmean=x0/2; ymean=y0/2;
        x=kron([-xmean:xmean-1],ones(y0,1));
        y=kron(ones(1,x0),[-ymean:ymean-1]');
        
        sigma=1;
        g1 = zeros(y0,x0); g2 = g1; g3 = g1;
        for i=1:y0
            for j=1:x0
                g1(i,j)=-exp(-((i-ymean)^2+(j-xmean)^2)/(2*sigma^2))*(i-ymean)/2/pi/sigma^2; % d/dy
                g2(i,j)=-exp(-((i-ymean)^2+(j-xmean)^2)/(2*sigma^2))*(j-xmean)/2/pi/sigma^2; % d/dx
                g3(i,j)= exp(-((i-ymean)^2+(j-xmean)^2)/(2*sigma^2))/2/pi/sigma^2;
            end
        end
        
        % fft2 二维离散傅里叶变换
        % ifft2 二维离散傅里叶反变换
        a=real(ifft2(fft2(f1).*fft2(g2))); % df1/dx, using circular convolution 循环卷积（圆周卷积）
        c=real(ifft2(fft2(f1).*fft2(g1))); % df1/dy, using circular convolution
        b=real(ifft2(fft2(f1).*fft2(g3)))-real(ifft2(fft2(f0).*fft2(g3))); % f1-f0
        R=c.*x-a.*y; % df1/dy*x-df1/dx*y
        
        a11 = sum(sum(a.*a)); a12 = sum(sum(a.*c)); a13 = sum(sum(R.*a));
        a21 = sum(sum(a.*c)); a22 = sum(sum(c.*c)); a23 = sum(sum(R.*c)); 
        a31 = sum(sum(R.*a)); a32 = sum(sum(R.*c)); a33 = sum(sum(R.*R));
        b1 = sum(sum(a.*b)); b2 = sum(sum(c.*b)); b3 = sum(sum(R.*b));
        Ainv = [a11 a12 a13; a21 a22 a23; a31 a32 a33]^(-1);
        s = Ainv*[b1; b2; b3];
        st = s;
        
        it=1;
        while ((abs(s(1))+abs(s(2))+abs(s(3))*180/pi/20>0.1)&it<25)
            % first shift and then rotate, because we treat the reference image
            % 先进行位移然后旋转，因为我们将运用参考图像
            f0_ = shift(f0,-st(1),-st(2));
            f0_ = imrotate(f0_,-st(3)*180/pi,'bicubic','crop');
            % A=imrotate（A，angle,' 旋转实现的方法'，'BBox'）;
            % bicubic 三次卷积插值法 
            % crop 剪切 旋转后超过图片原始大小的部分crop掉
            b = real(ifft2(fft2(f1).*fft2(g3)))-real(ifft2(fft2(f0_).*fft2(g3)));
            s = Ainv*[sum(sum(a.*b)); sum(sum(c.*b)); sum(sum(R.*b))];
            st = st+s;
            it = it+1;
        end
        % it
        
        st(3)=-st(3)*180/pi;
        st = st';
        st(1:2) = st(2:-1:1);
        stot = [2*stot(1:2)+st(1:2) stot(3)+st(3)];
        if pyrlevel>1
            % first rotate and then shift, because this is cancelling the
            % motion on the image to be registered
            %先旋转后位移 因为取消了图像的配准（取消了将要配准图像的动作？）
            im1{pyrlevel-1} = imrotate(im1{pyrlevel-1},-stot(3),'bicubic','crop');
            im1{pyrlevel-1} = shift(im1{pyrlevel-1},2*stot(2),2*stot(1)); 
            % twice the parameters found at larger scale
            % 两次发现较大规模的参数
        end
    end
    phi_est(imnr) = stot(3);
    delta_est(imnr,:) = stot(1:2);
end