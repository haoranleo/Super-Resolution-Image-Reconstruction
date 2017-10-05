function [ upper_left, upper_right, lower_left, lower_right ] = Decimate4( complete_image )
%DECIMATE4 Summary of this function goes here
%  Detailed explanation goes here

% complete_image is actually a MOVIE!!


for o=1:size(complete_image,2)
    upper_left(o).cdata=complete_image(o).cdata(1:2:end,1:2:end,:);
    upper_left(o).colormap=complete_image(o).colormap;
    
    upper_right(o).cdata=complete_image(o).cdata(1:2:end,2:2:end,:);
    upper_right(o).colormap=complete_image(o).colormap;
    
    lower_left(o).cdata=complete_image(o).cdata(2:2:end,1:2:end,:);
    lower_left(o).colormap=complete_image(o).colormap;
    
    lower_right(o).cdata=complete_image(o).cdata(2:2:end,2:2:end,:);
    lower_right(o).colormap=complete_image(o).colormap;
end

