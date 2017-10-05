function [ val ] = trng( x )
%TRNG Summary of this function goes here
%  Detailed explanation goes here
val=rect(x).*(1-2*abs(x));