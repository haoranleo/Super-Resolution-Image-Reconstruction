function [ val ] = gau( x ,sig )
%  Detailed explanation goes here

val=(1/sqrt(2*pi*sig^2)).*exp(-x.^2/(2*sig^2));
% if (x<-1/2 | x>1/2)
%     val=0;
% else
%     val=1;
% end