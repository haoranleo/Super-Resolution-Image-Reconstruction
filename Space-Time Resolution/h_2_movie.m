function [ movie_new ] = h_2_movie( h, NumOfFrames, Width, Height )
%H_2_MOVIE Summary of this function goes here
%  Detailed explanation goes here

movie_new=[];
for k=1:NumOfFrames
    size(h(k:NumOfFrames:end))
    movie_new(k).cdata=transpose(reshape(h(k:NumOfFrames:end),Height,Width));
    movie_new(k).colormap=[];
end