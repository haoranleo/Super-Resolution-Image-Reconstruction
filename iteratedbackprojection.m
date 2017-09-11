function [I Frames] = iteratedbackprojection(s, delta_est, phi_est, factor)
% ITERATEDBACKPROJECTION - Implementation of the iterated back projection SR algorithm
%                        -迭代投影法
%    s: images in cell array (s{1}, s{2},...)
%    delta_est(i,Dy:Dx) estimated shifts in y and x
%    delta_est用x和y坐标估计位移
%    phi_est(i) estimated rotation in reference to image number 1
%    phi_est利用各图像与第一幅图像的参照估计旋转
%    factor: gives size of reconstructed image
%            给出重建图像的大小

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
movieCounter = 1;
imOrigBig = imresize(s{1}, factor, 'nearest');
if(outputFrames)
    figure;
end
% -- End of Movie Variables

%% Initialization
lambda = 0.1; % define the step size for the iterative gradient method 
              % 定义迭代梯度法的步长
max_iter = 100;
iter = 1;

% Start with an estimate of our HR image: we use an upsampled version of
% the first LR image as an initial estimate.
% 从估计高分辨率图像开始：我们可以用第一幅低分辨率图像的没有采样过的版本作为初始的估计
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
%% Main loop
while iter < max_iter
    waitbar(min(10*iter/max_iter, 1), wait_handle);
    % Compute the gradient of the total squared error of reassembling the HR
    % 计算重组的高分辨率图像的梯度的总平方误差
    
    % image:
    %iter
    % --- Save each movie frame --- 保存每个电影帧？
    if(outputFrames)
        imshow(X);
        Frames(movieCounter) = getframe;
        movieCounter = movieCounter + 1;
    end
    % -----------------------------
    G = zeros(size(X));
    for i=1:length(s)
        temp = circshift(X, -[round(factor * delta_est(i,1)), round(factor * delta_est(i,2))]);
        temp = imrotate(temp, phi_est(i), 'crop');
        
        %temp = PSF * temp;
        temp = imfilter(temp, blur, 'symmetric');
        
        temp = temp(1:factor:end, 1:factor:end);
        temp = temp - s{i};
        temp = imresize(temp, factor, 'nearest');
        
        %temp = PSF' * temp;
        temp = imfilter(temp, sharpen, 'symmetric');
        
        temp = imrotate(temp, -phi_est(i), 'crop');
        G = G + circshift(temp, [round(factor * delta_est(i,1)), round(factor * delta_est(i,2))]);
    end

    % Now that we have the gradient, we will go in its direction with a step
    % 现在我们有了梯度，我们将在它的方向研究步长值
    % size of lambda  lanmda的值
    X = X - (lambda) * G;   
    %max(X(:))
    %max(G(:))
    %X = X / max(X(:));
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
close(wait_handle);
I = X;
