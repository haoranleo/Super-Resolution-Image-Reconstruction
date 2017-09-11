function delta_est = keren_shift(im)
% KEREN_SHIFT shift estimation using algorithm by Keren et al.
%             用Keren算法估计位移
%    delta_est = keren_shift(im)
%    shift estimation algorithm implemented from the paper by Keren et al.

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%

for imnr = 2:length(im)
    % construct pyramid scheme
    % 构建金字塔体
    
    lp = fspecial('ga',5,1);
    im0{1} = im{1};
    im1{1} = im{imnr};
    for i=2:3
        im0{i} = imresize(conv2(im0{i-1},lp,'same'),0.5,'bicubic');
        im1{i} = imresize(conv2(im1{i-1},lp,'same'),0.5,'bicubic');
    end
    
    stot = zeros(1,2);
    % do actual registration, based on pyramid
    % 根据金字塔阵配准图像
    
    for pyrlevel=3:-1:1
        f0 = im0{pyrlevel};
        f1 = im1{pyrlevel};
        
        [y0,x0]=size(f0);
        xmean=x0/2; ymean=y0/2;
        sigma=1;
        x=kron([-xmean:xmean-1],ones(y0,1));
        y=kron(ones(1,x0),[-ymean:ymean-1]');
        g1 = zeros(y0,x0); g2 = g1; g3 = g1;
        for i=1:y0
            for j=1:x0
                g1(i,j)=-exp(-((i-ymean)^2+(j-xmean)^2)/2)*(i-ymean)/2/pi; % d/dy
                g2(i,j)=-exp(-((i-ymean)^2+(j-xmean)^2)/2)*(j-xmean)/2/pi; % d/dx
                g3(i,j)=exp(-((i-ymean)^2+(j-xmean)^2)/2)/2/pi;
            end
        end
        
        a=real(ifft2(fft2(f1).*fft2(g2))); % df1/dx
        c=real(ifft2(fft2(f1).*fft2(g1))); % df1/dy
        b=real(ifft2(fft2(f1).*fft2(g3)))-real(ifft2(fft2(f0).*fft2(g3))); % f1-f0
        A=[a(:) , c(:)];
        s=lsqlin(A,b(:)); 
        stot=s;
        
        while (abs(s(1))+abs(s(2))>0.05)
            f0_ = shift(f0,-stot(1),-stot(2));
            b = real(ifft2(fft2(f1).*fft2(g3)))-real(ifft2(fft2(f0_).*fft2(g3)));
            s=lsqlin(A,b(:));
            stot = stot+s;
        end
        stot = stot';
        stot(1:2) = stot(2:-1:1);
    end
    delta_est(imnr,:) = -stot;
end
