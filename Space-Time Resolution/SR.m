
%
% Super Resolution Algorithm -  Direction of Use 
% ==============================================  
%
% In order to use the SR algorithm you have to go through these steps:
%
% 1. Choose the movie you want to work on. It can be in color, but the analysis is made
%    on the Y layer in the YCbCr representation. The movie will be used to synthetically 
%    creating input movies.
%
% 2. Create an Excel worksheet containing this information (see attached files for examples):
%       rate_t  =   The rate in which you want to down-sample the movie in the time axis
%       rate_x  =   Same as above, for the X axis
%       rate_y  =   Same as above, for the Y axis
%       shift_t =   The shift from the down-sampling starts, away from the first frame (0 is no-shift at all)
%       shift_x =   Same as above, for the X axis.
%       shift_y =   Same as above, for the Y axis.
%       func_t  =   Leave 1 (Always uniform blurring)
%       nop_t   =   Leave 2 (Two parameters follow)
%       params_t    =   As pointed, you should provide two parameters in this order:
%                          * Defines the size of the blurring in rate_t units (rate_t*param1 is the number
%                            of frames that are blurred together
%                          * Always put half the previous parameter.
%       func_x  =   1 - Rect, 2 - finite-support Gaussian
%       nop_x   =   for each function different number of parameters needed. see their documentation.
%       params_x    =   According to the function, provide the neccesary parameters.
%       func_y, nop_y, params_y = Same as for X
%
%   3. Fill in the name of the movie in the line "movie_name='YourMovieNameGoesHere.avi'" (Uncompressed AVI)
%
%   4. Fill in the name of your Excel worksheet in the line "parameters=xlsread('...')"
%
%   5. Define the Resol_t, Resol_x, Resol_y parameters, which are the wanted resoultion in each axis.
%       NOTE: their product should be smaller than the total number of points in all the input movies.
%
%   6. The parameter "nof" can limit the number of frames taken from the movie.
% 
%   7. Fill in the desired output movie name's prefix and .mat where results are stored. (In this file)
%
%   8. After activating SR.m, you will get in your chosen .mat file the A matrix, L vector
%       of the input from all the created movies and resolution parameters of the final movie.
%       Solve using lsqlin in Matlab.
%
%   9. You get the vector H, containing the output movie, in a vector, in t-x-y order.

clear all
close all

out_prefix='spacetime1_';

movie_name='bounce.avi';

movie_in=aviread(movie_name);
info=aviinfo(movie_name);


Resol_t=10;
Resol_x=54;
Resol_y=75;

function_names={'rect' 'gau'};
k=1;
nom=0;
parameters=xlsread('spacetime1.xls');
while k<=size(parameters,1)
    
    if isnan(parameters(k,1))==1
        k=k+1;
        continue
    end
    
    nom=nom+1;
    input{nom}.movie_number=nom;
    u=0;
    
    u=u+1;    input{nom}.rate_t=parameters(k,u);
    u=u+1;    input{nom}.rate_x=parameters(k,u);
    u=u+1;    input{nom}.rate_y=parameters(k,u);  
    
    u=u+1;    input{nom}.shift_t=parameters(k,u);    
    u=u+1;    input{nom}.shift_x=parameters(k,u);
    u=u+1;    input{nom}.shift_y=parameters(k,u);  
    
    u=u+1;    input{nom}.func_t=str2func(function_names(parameters(k,u)));
    u=u+1;    input{nom}.parameters_t=parameters(k,u+1:u+parameters(k,u));
    u=u+parameters(k,u);
    while isnan(parameters(k,u+1))==1
        u=u+1;
    end
    
    u=u+1;    input{nom}.func_x=str2func(function_names(parameters(k,u)));
    u=u+1;    input{nom}.parameters_x=parameters(k,u+1:u+parameters(k,u));
    u=u+parameters(k,u);
    while isnan(parameters(k,u+1))==1
        u=u+1;
    end
    
    u=u+1;    input{nom}.func_y=str2func(function_names(parameters(k,u)));
    u=u+1;    input{nom}.parameters_y=parameters(k,u+1:u+parameters(k,u));
    u=u+parameters(k,u);
    while (u<size(parameters(k,2)) & isnan(parameters(k,u+1))==1)
        u=u+1;
    end    
    k=k+1;
end


% Creating the movies
nof= 59;    %info.NumFrames;
movie_new=[];
for k=1:nof 
    movie_new(k).cdata=movie_in(k).cdata(:,1:72,:);     % Choose the part of the frame that will be taken (x,y range. 3 axis leave :)
    movie_new(k).colormap=movie_in(k).colormap;
end
clear movie_in;
movie_in=movie_new;

% Converting to Black and White (Y Layer)
movie_new=[];
for f=1:nof
    movie_in(f).cdata=im2double(movie_in(f).cdata);
    
    [movie_in(f).cdata(:,:,1), movie_in(1).cdata(:,:,2), movie_in(f).cdata(:,:,3)]=...
        rgb2ycbcr(movie_in(f).cdata(:,:,1), movie_in(f).cdata(:,:,2), movie_in(f).cdata(:,:,3));
    
    movie_new(f).cdata=im2double(movie_in(f).cdata(:,:,1));    % Taking the Y layer only
    movie_new(f).colormap=movie_in(f).colormap; 
    
end
movie_in=movie_new;

movies=cell(1,nom);


% Spatial blurring each movie
for k=1:nom
    ker_size_x=2*ceil(input{k}.rate_x*(input{k}.parameters_x(1)/2))+1;
    ker_size_y=2*ceil(input{k}.rate_y*(input{k}.parameters_y(1)/2))+1;
    
    [X,Y]=meshgrid([-(ker_size_x-1)/2:(ker_size_x-1)/2]/input{k}.rate_x,[-(ker_size_y-1)/2:(ker_size_y-1)/2]/input{k}.rate_y);
    
    spatial_blur_kernel=feval(input{k}.func_x,X,input{k}.parameters_x).*feval(input{k}.func_y,Y,input{k}.parameters_y);
    spatial_blur_kernel=spatial_blur_kernel/sum(spatial_blur_kernel(:));
    movies{k}=spatialblur(movie_in, spatial_blur_kernel);
end

movies2=movies;

% Temporal blurring and sampling of each movie
for k=1:nom
    
    start_from_frame=input{k}.shift_t*input{k}.rate_t + 1;
    sampling_rate=input{k}.rate_t;
    blur_length=input{k}.parameters_t(1)*input{k}.rate_t;

    movies{k}=TemporalBlur(movies{k}, start_from_frame, sampling_rate, blur_length);
    
end



% spatial sampling of each movie
for k=1:nom
    for j=1:length(movies{k})
        movies{k}(j).cdata=movies{k}(j).cdata( (input{k}.shift_y*input{k}.rate_y+1):input{k}.rate_y:end, (input{k}.shift_x*input{k}.rate_x+1):input{k}.rate_x:end);
    end
end


movies_names=[];

% Exporting the movies in uncompressed avi
fps_saved=5;
movies_output=movies;
for k=1:nom
    for j=1:length(movies{k})
        movies_output{k}(j).cdata(:,:,1)=movies{k}(j).cdata;
        movies_output{k}(j).cdata(:,:,2)=movies{k}(j).cdata;        
        movies_output{k}(j).cdata(:,:,3)=movies{k}(j).cdata;
        movies_output{k}(j).colormap=[];
    end
    
    eval('movname=[out_prefix num2str(k)];');
    movies_names=[movies_names {movname}];
    
    movie2avi(movies_output{k},movname,'FPS',fps_saved,'COMPRESSION','None');
end

% Creating L from the movies.
L=[];
for m=1:nom
    temp_mat=zeros(size(movies{m}(1).cdata(:,:,1),1),size(movies{m}(1).cdata(:,:,1),2),length(movies{m}));
    
    for f=1:length(movies{m})
        temp_mat(:,:,f)=movies{m}(f).cdata(:,:,1);
    end
    
    temp_mat=permute(temp_mat,[3,2,1]);
    
    L=[L transpose(temp_mat(:)) ];
end


name1='spacetime1.mat';
delete 'spacetime1.mat'
save 'spacetime1.mat' L
save 'spacetime1.mat' Resol_x -append
save 'spacetime1.mat' Resol_y -append
save 'spacetime1.mat' Resol_t -append

clear L;
clear movie_in;
clear movie_new;
clear movies_output;
clear movies;

T_Mat=TMatConstruction(movies_names,input,Resol_x,Resol_y,Resol_t);
A=Aconstruction7( movies_names,Resol_x,Resol_y,Resol_t,T_Mat,input);

save 'spacetime1.mat' A -append