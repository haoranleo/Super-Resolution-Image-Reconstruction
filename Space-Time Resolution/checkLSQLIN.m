%CHECKLSQLIN Summary of this function goes here
%  Detailed explanation goes here
clear all;
times=[];
op=[20 50 100 150];

for k=1:length(op)
    A=ceil(rand(10*op(k),10*op(k)/2)*5);
    L=rand(1,10*op(k));
    tic;
    X=lsqlin(A,L,[],[]);
    times=[times toc];
end
    