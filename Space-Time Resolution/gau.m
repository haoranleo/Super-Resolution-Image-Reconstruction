function [ val ] = gau( x ,params )
%  GAU  ( x , [ sup shift sigma ] )
%
%   x ~ Gaussian(mean=shift, var=sigma^2) (Normalized to unity integral),       if -0.5<=((x-shift)/sup)<0.5
%       0,                                       else                   
%


sup=params(1);
shift=params(2);
sigma=params(3);

val=(1/(Q(-sup/2)-Q(sup/2)))*(1/sqrt(2*pi*sigma^2)).*exp(-(x-shift).^2/(2*sigma^2)) .* (-1/2 <= (x-shift)/sup  &  (x-shift)/sup < 1/2);
