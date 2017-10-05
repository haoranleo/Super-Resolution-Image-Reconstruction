close all
clear all

movie_name='uncompressed.avi';

movie_in=aviread(movie_name);
info=aviinfo(movie_name);


movie_new=[];
for k=1:info.NumFrames
%movie_new(k).cdata=movie_in(k).cdata(1:info.Height/2,1:info.Width/2,:);
movie_new(k).cdata=movie_in(k).cdata(1:16,1:16,:);
movie_new(k).colormap=movie_in(k).colormap;
end
clear movie_in;
movie_in=movie_new;


full_movie_blurred=TemporalBlur(movie_in,1,105,5);
full_movie_blurred=spatialblur(full_movie_blurred, ones(3,3)/9);



[mul,mur,mll,mlr]=decimate4(full_movie_blurred);

stam=meshgrid(1:128);
stam=0.5*stam+0.5*stam';
stam=stam/max(stam(:));


stam(:,:)=0.5;
% stam(1,:)=0;
% stam(:,1)=0;
% stam(:,end)=0;
% stam(end,:)=0;

%stam(end/2-1:end/2+1,end/2-1:end/2+1)=1;
%stam=filter2(ones(3,3)*(1/9),stam,'same');

stm(:,:,1)=stam;%filter2( ones(5,5)/25, stam);
stm(:,:,2)=stam;%filter2( ones(5,5)/25, stam);
stm(:,:,3)=stam;%filter2( ones(5,5)/25, stam);
size(stm)

mul(1).cdata=stm(1:2:end,1:2:end,:);
mur(1).cdata=stm(1:2:end,2:2:end,:);
mll(1).cdata=stm(2:2:end,1:2:end,:);
mlr(1).cdata=stm(2:2:end,2:2:end,:);



figure;
subplot(2,2,1)
imshow(mul(1).cdata)

subplot(2,2,2)
imshow(mur(1).cdata)
subplot(2,2,3)
imshow(mll(1).cdata)
subplot(2,2,4)
imshow(mlr(1).cdata)
fps_saved=5;
figure;
imshow(stm);



movie2avi(full_movie_blurred,'full_movie_blurred.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mul,'mul.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mur,'mur.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mll,'mll.avi','FPS',fps_saved,'COMPRESSION','None');
movie2avi(mlr,'mlr.avi','FPS',fps_saved,'COMPRESSION','None');