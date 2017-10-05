close all
crs=1000*2.^[0:7];

times=[];

for i=1:length(crs)
tic
vec1=ones(crs(i),1);
A=spdiags(vec1,0,crs(i),crs(i));
A=kron(A,[1 1 1 0]);
vec2=kron(ones(crs(i)/100,1),[1:100]');
size(A)

size(vec2)

t=A\vec2;
times=[times toc];
end


figure;
plot(crs,times);
