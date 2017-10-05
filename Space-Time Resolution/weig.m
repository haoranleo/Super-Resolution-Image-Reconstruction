A=A/max(A(:));
weig=sum(A,2);
for r=1:1024
    A(r,:)=A(r,:)/weig(r);
end