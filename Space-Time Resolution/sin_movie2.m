clear all
close all

ball=imread('ball.bmp');
mov=[];
frame=0;
for t=10:189
    frame=frame+1;
    mv=zeros(100,200);
    y=50+round(40*sin(((2*pi)/90)*t));
    mv(y-7:y+7,t-7:t+7)=ball;
    mov(frame).cdata(:,:,1)=uint8(mv);
    mov(frame).cdata(:,:,2)=uint8(mv);
    mov(frame).cdata(:,:,3)=uint8(mv);
    mov(frame).colormap=[];
end
    
      