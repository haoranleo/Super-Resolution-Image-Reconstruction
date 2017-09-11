function rec = papoulisgerchberg(s,delta_est,factor)
% PAPOULISGERCHBERG - reconstruct high resolution image using Papoulis Gerchberg algorithm
%                     用Papoulis Gerchberg算法重建高分辨率图像
%    rec = papoulisgerchberg(s,delta_est,factor)
%    reconstruct an image with FACTOR times more pixels in both dimensions
%    using Papoulis Gerchberg algorithm and using the shift and rotation 
%    information from DELTA_EST and PHI_EST
%    in:
%    s: images in cell array (s{1}, s{2},...)
%    delta_est(i,Dy:Dx) estimated shifts in y and x
%    delta_est估计x和y方向的位移
%    factor: gives size of reconstructed image
%    factor规定了重建图片的大小

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%
max_iter = 25;

temp = upsample(upsample(s{1}, factor)', factor)';
y = zeros(size(temp));
coord = find(temp);
y(coord) = temp(coord);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% construction of zero_cols, zero_rows 
% factor_size_FFT is introduced in case that
% in PG2D the size of the FFT2 is increased
% fft2(y,factor_size_FFT*N(1),factor_size_FFT*N(2))
% !!! see if borders to be included or not!!!
NHR = size(temp);
NLR = floor(sqrt(length(s))*size(s{1})/2)*2;
zero_rows = (1+NLR(1)/2+1)-1:(NHR(1)-NLR(1)/2-1)+1;
zero_cols = (1+NLR(2)/2+1)-1:(NHR(2)-NLR(2)/2-1)+1;


for i = length(s):-1:1
    temp = upsample(upsample(s{i}, factor)', factor)';
    temp = shift(temp, round(delta_est(i, 2)*factor), round(delta_est(i, 1)*factor));
    coord = find(temp);
    y(coord) = temp(coord);
end
   
y_prev=y;

E=[];
iter=1;


wait_handle = waitbar(0, 'Reconstruction...', 'Name', 'SuperResolution GUI');
while iter < max_iter
   waitbar(min(4*iter/max_iter, 1), wait_handle);
   Y=fft2(y);
   Y(zero_rows,:)=0;
   Y(:,zero_cols)=0;
   y=ifft2(Y);
   
   for i = length(s):-1:1
        temp = upsample(upsample(s{i}, factor)', factor)';
        temp = shift(temp, round(delta_est(i, 2)*factor), round(delta_est(i, 1)*factor));
        coord = find(temp);
        y(coord) = temp(coord);
   end
   
   delta= norm(y-y_prev)/norm(y);
   E=[E; iter delta];
   iter= iter+1;
   if iter>3 
     if abs(E(iter-3,2)-delta) <1e-8
        break  
     end
   end
   y_prev=y;
%    if mod(iter,10)==2
%        disp(['iteration ' int2str(E(iter-1,1)) ', error ' num2str(E(iter-1,2))])
%    end
end

close(wait_handle);
% reconstructed image
rec = real(y);