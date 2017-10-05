lin=size(A,1);

t=0;
for r=1:lin-1
    for g=r+1:lin
g
        if A(r,:)==A(g,:)

            t=t+1;
            [r,g]
        end
    end
end
