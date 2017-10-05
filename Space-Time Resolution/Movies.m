close all
clear all

movie_name='uncompressed.avi';

movie_in=aviread(movie_name);
info=aviinfo(movie_name);

% Movie size modification

new_size_s=32;
new_size_e=95;

movie_new=[];
for k=1:info.NumFrames
    movie_new(k).cdata=movie_in(k).cdata(84:147,1:96,:);
    movie_new(k).colormap=movie_in(k).colormap;
end
clear movie_in;
movie_in=movie_new;




movie_new=[];
for f=1:info.NumFrames
    movie_in(f).cdata=im2double(movie_in(f).cdata);
    
    [movie_in(f).cdata(:,:,1), movie_in(1).cdata(:,:,2), movie_in(f).cdata(:,:,3)]=...
        rgb2ycbcr(movie_in(f).cdata(:,:,1), movie_in(f).cdata(:,:,2), movie_in(f).cdata(:,:,3));
    
    movie_new(f).cdata=movie_in(f).cdata(:,:,1);    % Taking the Y layer only
    
    imshow(movie_new(f).cdata);    
end





spatial_blur_kernel=ones(3,3)/9; %[0 0 0; 0 1 0; 0 0 0];
start_from_frame=1;
sampling_rate=105;
blur_length=1;


full_movie_blurred=TemporalBlur(movie_in,start_from_frame,sampling_rate,blur_length);
full_movie_blurred=spatialblur(full_movie_blurred, spatial_blur_kernel);


[mul,mur,mll,mlr]=decimate4(full_movie_blurred);


fps_saved=5;

movie2avi(full_movie_blurred,'full_movie_blurred.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mul,'mul.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mur,'mur.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mll,'mll.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mlr,'mlr.avi','FPS',fps_saved,'COMPRESSION','None');


L=[];
nom=4;

movies=cell(1,nom);
movies{1}=mul;
movies{2}=mur;
movies{3}=mll;
movies{4}=mlr;

for m=1:nom
    temp_mat=zeros(size(movies{m}(1).cdata(:,:,1),1),size(movies{m}(1).cdata(:,:,1),2),length(movies{m}));
    
    for f=1:length(movies{m})
        temp_mat(:,:,f)=movies{m}(f).cdata(:,:,1);
    end
    
    temp_mat=permute(temp_mat,[3,2,1]);
    
    L=[L transpose(temp_mat(:)) ];
end