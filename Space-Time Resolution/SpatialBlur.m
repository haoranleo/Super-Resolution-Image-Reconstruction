function [ movie_out ] = SpatialBlur( movie_in, filt )
% blur each frame in the movie with the specifed filter (filt)

total_frames = size(movie_in,2);
size_of_frame=size(movie_in(1).cdata);


for i=1:total_frames
    current_in_frame=double(movie_in(i).cdata);
    for r=1:size(current_in_frame,3)
        current_out_frame(:,:,r)=filter2(filt,current_in_frame(:,:,r),'same');

    end
    movie_out(i).cdata=current_out_frame;
    movie_out(i).colormap = movie_in(i).colormap;
end
