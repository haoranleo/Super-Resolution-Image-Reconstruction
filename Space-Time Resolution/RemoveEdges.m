size_of_h=sqrt(size(A,2));
p1=[1:size_of_h];
p2=[size_of_h^2-size_of_h+1:size_of_h^2];
p3=[1:size_of_h:size_of_h^2-size_of_h+1];
p4=[size_of_h:size_of_h:size_of_h^2];

p=setdiff(1:size_of_h^2,[p1 p2 p3 p4]);

AA=spalloc(size(A,1),length(p),length(find(A)));
for g=1:length(p)
    AA(:,g)=A(:,p(g));
end

