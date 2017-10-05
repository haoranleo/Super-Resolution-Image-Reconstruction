clear all
close all

figure;
ball=imread('ball150.bmp');
mov=[];
frame=0;
for t=10:0.1:189
    frame=frame+1;
    mv=zeros(1000,2000);
    y=500+round(400*sin(((2*pi)/90)*t));
    mv(y-74:y+75,10*t-74:10*t+75)=ball;
    mov(frame).cdata(:,:,1)=mv;
    mov(frame).cdata(:,:,2)=mv;
    mov(frame).cdata(:,:,3)=mv;
    mov(frame).colormap=[];
end
    
      