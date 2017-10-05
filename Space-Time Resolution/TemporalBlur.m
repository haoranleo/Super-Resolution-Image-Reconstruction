function [ movie_out ] = TemporalBlur( movie_in, start_from_frame, sampling_rate, blur_length )
% Temporal Blur of the movie. starting from the start_from_frame frame, each group of succesive
%   blur_length frames are blurred together. next group beginning is after sampling_rate frames.

total_frames = size(movie_in,2);
size_of_frame=size(movie_in(1).cdata);

last_frame = floor((total_frames-start_from_frame)/sampling_rate)*sampling_rate+start_from_frame

frames_vector=[start_from_frame:sampling_rate:last_frame]



if (total_frames - last_frame +1 < blur_length) 
    frames_vector=frames_vector(1:end-(1+floor(blur_length/sampling_rate)))
end


for i=1:length(frames_vector)

    movie_out(i).cdata=zeros(size_of_frame);

    
    for j=0:blur_length-1
        movie_out(i).cdata = movie_out(i).cdata + movie_in(frames_vector(i)+j).cdata;
    end
    
    movie_out(i).cdata=(1/blur_length)*movie_out(i).cdata;
    
    movie_out(i).colormap = movie_in(frames_vector(i)).colormap;
end

