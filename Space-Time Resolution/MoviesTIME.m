close all
clear all

movie_name='uncompressed.avi';

nom=3;

movies=cell(1,nom);

movie_in=aviread(movie_name);
info=aviinfo(movie_name);

% Movie size modification

new_size_s=32;
new_size_e=95;
nof=30;

movie_new=[];
for k=1:nof % info.NumFrames
    movie_new(k).cdata=movie_in(k).cdata(84:147,1:64,:);
    movie_new(k).colormap=movie_in(k).colormap;
end
clear movie_in;
movie_in=movie_new;




movie_new=[];
for f=1:nof
    movie_in(f).cdata=im2double(movie_in(f).cdata);
    
    [movie_in(f).cdata(:,:,1), movie_in(1).cdata(:,:,2), movie_in(f).cdata(:,:,3)]=...
        rgb2ycbcr(movie_in(f).cdata(:,:,1), movie_in(f).cdata(:,:,2), movie_in(f).cdata(:,:,3));
    
    movie_new(f).cdata=movie_in(f).cdata(:,:,1);    % Taking the Y layer only
    movie_new(f).colormap=movie_in(f).colormap; 
    
    imshow(movie_new(f).cdata);    
end
movie_in=movie_new;

spatial_blur_kernel=[0 0 0; 0 1 0; 0 0 0];
movie_in=spatialblur(movie_in, spatial_blur_kernel);




sff=[1 2 3];
for j=1:3

start_from_frame=sff(j);
sampling_rate=7;
blur_length=3;

movies{j}=TemporalBlur(movie_in,start_from_frame,sampling_rate,blur_length);

end

fps_saved=5;

movie2avi(movie_in,'full_movie_blurred.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(movies{1},'m1.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(movies{2},'m2.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(movies{3},'m3.avi','FPS',fps_saved,'COMPRESSION','None');


L=[];




for m=1:nom
    temp_mat=zeros(size(movies{m}(1).cdata(:,:,1),1),size(movies{m}(1).cdata(:,:,1),2),length(movies{m}));
    
    for f=1:length(movies{m})
        temp_mat(:,:,f)=im2double(movies{m}(f).cdata(:,:,1));
    end
    
    temp_mat=permute(temp_mat,[3,2,1]);
    
    L=[L transpose(temp_mat(:)) ];
end