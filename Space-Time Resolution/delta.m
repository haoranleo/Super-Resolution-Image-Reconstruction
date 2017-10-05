function [val] = delta(x)
%DELTA Summary of this function goes here
%  Detailed explanation goes here
val=1;
for i=1:length(x)
    if ~(x(i)==0)
        val=0;
        break;
    end
end
