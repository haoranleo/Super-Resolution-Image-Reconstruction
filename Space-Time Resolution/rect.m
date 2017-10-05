function [ val ] = rect( x ,params )

%   RECT ( x , [ sup shift ] )
%   
%   x = 1/sup,      if -0.5<=((x-shift)/sup)<0.5
%       0,          else
%

sup=params(1);
shift=params(2);
val=(1/sup) * (-1/2 <= (x-shift)/sup  &  (x-shift)/sup < 1/2);
