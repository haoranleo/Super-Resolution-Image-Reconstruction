%clear all
close all

% % % movies_to_read=[{'mul.avi','mur.avi','mll.avi','mlr.avi'}];
% % % 
% % % movies=cell(1,length(movies_to_read));
% % % for j=1:length(movies_to_read)
% % %     movies{j}=aviread(char(movies_to_read(j)));
% % % end
movies_to_read=zeros(1,4);
movies=cell(1,length(movies_to_read));
movies{1}=mul;
movies{2}=mur;
movies{3}=mll;
movies{4}=mlr;

Y_Movies=cell(1,length(movies_to_read));

L=[];

temp_mat=[];

for m=1:length(movies_to_read)
    for f=1:length(movies{m})
        Y_Movies{m}(f).cdata=im2double(movies{m}(f).cdata);

         [Y_Movies{m}(f).cdata(:,:,1), Y_Movies{m}(1).cdata(:,:,2), Y_Movies{m}(f).cdata(:,:,3)]=...
             rgb2ycbcr(Y_Movies{m}(f).cdata(:,:,1), Y_Movies{m}(f).cdata(:,:,2), Y_Movies{m}(f).cdata(:,:,3));
        
        %imshow(Y_Movies{m}(f).cdata(:,:,1));    
    end

end


for m=1:length(movies_to_read)
    temp_mat=zeros(size(Y_Movies{m}(1).cdata(:,:,1),1),size(Y_Movies{m}(1).cdata(:,:,1),2),length(Y_Movies{m}));
    for f=1:length(Y_Movies{m})
        temp_mat(:,:,f)=Y_Movies{m}(f).cdata(:,:,1);
    end
    temp_mat=permute(temp_mat,[3,2,1]);
    L=[L transpose(temp_mat(:)) ];
end


% >> a1=kron(diag(sparse(ones(1,185472))),[1 1 1 0]);
% >> A=[a1;a1;a1;a1];