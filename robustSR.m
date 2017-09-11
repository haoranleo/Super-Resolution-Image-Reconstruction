function [I Frames] = robustSR(s, delta_est, phi_est, factor)
% ROBUSTSR - Implementation of a robust superresolution technique from Assaf Zomet, Alex Rav-Acha and Shmuel Peleg 
% ROBUSTSR - 根据艾瑟夫巴德Zomet Alex Rav-Acha Shmuel法勒实现一个完整的超限分辨技术

%    s: images in cell array (s{1}, s{2},...)
%    delta_est(i,Dy:Dx) estimated shifts in y and x
%    phi_est(i) estimated rotation in reference to image number 1       在参考图像估计旋转1号
%    factor: gives size of reconstructed image
%    因素:使重建图像的大小

%% -----------------------------------------------------------------------
% SUPERRESOLUTION - 超分辨率图像重建图形用户界面
% Copyright (C) 2016 Laboratory of Zhejiang University 
% UPDATED From Laboratory of Audiovisual Communications (LCAV)
%
if nargout > 1
    outputFrames = true;
else
    outputFrames = false;
end

%% Movie variables
% 电影变量
movieCounter = 1;
imOrigBig = imresize(s{1}, factor, 'nearest');  % 对图像进行缩放
if(outputFrames)
    figure;
end
% -- End of Movie Variables

%% Initialization
% 初始化
lambda = 0.05; % define the step size for the iterative gradient method
               % 定义迭代梯度法的步长
max_iter = 50;
iter = 1;

% Start with an estimate of our HR image: we use an upsampled version of
% the first LR image as an initial estimate.
% 首先估计我们的高分辨率图片:我们使用第一个低分辨率图片的未取样版本作为初始估计

X = imOrigBig;
X_prev = X;
E = [];

%imshow(X);

%PSF = generatePSF([1 0 0], [1 2 1], X);
blur = [0 1 0;...
        1 2 1;...
        0 1 0];
blur = blur / sum(blur(:));

sharpen = [0 -0.25 0;...
          -0.25 2 -0.25;...
           0 -0.25 0];

wait_handle = waitbar(0, 'Reconstruction...', 'Name', 'SuperResolution GUI');

while iter < max_iter
    waitbar(min(5*iter/max_iter, 1), wait_handle);
    % Compute the gradient of the total squared error of reassembling the HR
    % 计算估计的高分辨率图片的梯度总平方误差
    % image:
    % --- Save each movie frame ---
    % --- 保存每一个帧 ---
    if(outputFrames)
        imshow(X); title(num2str(iter));
        Frames(movieCounter) = getframe;
        movieCounter = movieCounter + 1;
    end
    % -----------------------------
    for i=1:length(s)
        temp = circshift(X, -[round(factor * delta_est(i,1)), round(factor * delta_est(i,2))]);
        % circshift 循环位移函数
        temp = imrotate(temp, phi_est(i), 'crop');
        % imrotate 旋转图像
        %temp = PSF * temp;
        temp = imfilter(temp, blur, 'symmetric');
        % imfilter 　B = imfilter(A,H,option1,option2,...)
        % 或写作g = imfilter(f, w, filtering_mode, boundary_options, size_options)
        % 其中，f为输入图像，w为滤波掩模，g为滤波后图像。filtering_mode用于指定在滤波过程中是使用“相关”还是“卷积”。4
        % boundary_options用于处理边界充零问题，边界的大小由滤波器的大小确定。

        temp = temp(1:factor:end, 1:factor:end);
        temp = temp - s{i};
        temp = imresize(temp, factor, 'nearest');
        
        %temp = PSF' * temp;
        temp = imfilter(temp, sharpen, 'symmetric');
        
        temp = imrotate(temp, -phi_est(i), 'crop');
        G(:,:,i) = circshift(temp, [round(factor * delta_est(i,1)), round(factor * delta_est(i,2))]);
    end
    % Take the median of G, element by element
    % 取G的中值作为G
    M = median(G, 3);
    % Now that we have the median, we will go in its direction with a step
    % 现在我们有中值,我们将会在它的方向迈出的一步
    % size of lambda
    % λ的大小

    X = X - length(s)*lambda * M;   
   
    delta = norm(X-X_prev)/norm(X);
    E=[E; iter delta];
    if iter>3 
      if abs(E(iter-3,2)-delta) <1e-4
         break  
      end
    end
    X_prev = X;
    iter = iter+1;
   
end

disp(['Ended after ' num2str(iter) ' iterations.']);
disp(['Final error is ' num2str(abs(E(iter-3,2)-delta)) ' .']);
%figure;
%imshow(X);
I = X;
close(wait_handle);