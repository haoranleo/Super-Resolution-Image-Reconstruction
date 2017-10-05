function [y,cb,cr]=rgb2ycbcr(r,g,b)
%
% [y,cb,cr]=rgb2ycbcr(r,g,b)
% 
% Color conversion. From RGB to YCbCr.
% The R,G and B components must be of the same 
% vector(matrix) size.
%
r=r*256;
g=g*256;
b=b*256;
  A=[65.738 129.057 25.06;-37.945 -74.494 112.439;112.439 -94.154 ...
	 -18.285]/256;
  B=[16;128;128];
  

  [v,h]=size(r);
  r=r(:)';
  g=g(:)';
  b=b(:)';
  xc=[r;g;b];
  yc=A*xc+B*ones(1,length(r));
  
  y=round(yc(1,:));
  cb=round(yc(2,:));
  cr=round(yc(3,:));
  
  y=y.*(y>=16 & y<=235)+16*(y<16)+235*(y>235);
  cb=cb.*(cb>=16 & cb<=240)+16*(cb<16)+240*(cb>240);
  cr=cr.*(cr>=16 & cr<=240)+16*(cr<16)+240*(cr>240);
  
  y=(reshape(y,[v h]))/256;
  cb=(reshape(cb,[v h]))/256;
  cr=(reshape(cr,[v h]))/256;
  