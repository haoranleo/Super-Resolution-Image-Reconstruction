function y = pocs(s,delta_est,factor)
% POCS - reconstruct high resolution image using Projection On Convex Sets
%        用凸集投影法重建高分辨率图像
%    y = pocs(s,delta_est,factor)
%    reconstruct an image with FACTOR times more pixels in both dimensions
%    using Papoulis Gerchberg algorithm and using the shift and rotation 
%    information from DELTA_EST and PHI_EST
%    利用DELTA_EST和PHI_EST携带的位移和旋转信息以及采用PG算法重建具有更多像素点的图像
%    in:
%    s: images in cell array (s{1}, s{2},...)
%    delta_est(i,Dy:Dx) estimated shifts in y and x
%    factor: gives size of reconstructed image

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%
max_iter = 50;

temp = upsample(upsample(s{1}, factor)', factor)';
y = zeros(size(temp));
coord = find(temp);
y(coord) = temp(coord);


for i = 2:length(s)
    temp = upsample(upsample(s{i}, factor)', factor)';
    temp = shift(temp, round(delta_est(i, 2)*factor), round(delta_est(i, 1)*factor));
    coord = find(temp);
    y(coord) = temp(coord);
end
   
y_prev=y;

E=[];
iter=1;

blur =[.25 0 1 0 .25;...
        0  1 2 1  0;...
        1  2 4 2  1;...
        0  1 2 1  0;...
       .25 0 1 0 .25];
   
blur = blur / sum(blur(:));
wait_handle = waitbar(0, 'Reconstruction...', 'Name', 'SuperResolution GUI');

while iter < max_iter
   waitbar(min(4*iter/max_iter, 1), wait_handle);
   y = imfilter(y, blur);   
   for i = length(s):-1:1
        temp = upsample(upsample(s{i}, factor)', factor)';
        temp = shift(temp, round(delta_est(i, 2)*factor), round(delta_est(i, 1)*factor));
        coord = find(temp);
        y(coord) = temp(coord);
   end
   
   delta= norm(y-y_prev)/norm(y);
   E=[E; iter delta];
   iter = iter+1;
   if iter>3 
     if abs(E(iter-3,2)-delta) <1e-4
        break  
     end
   end
   y_prev=y;
%    if mod(iter,10)==2
%        disp(['iteration ' int2str(E(iter-1,1)) ', error ' num2str(E(iter-1,2))])
%    end
end

close(wait_handle);