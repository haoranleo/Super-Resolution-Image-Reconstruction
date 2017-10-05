stam=meshgrid(1:32);
stam=0.5*stam+0.5*stam';
stam=stam/max(stam(:));

[X,Y]=meshgrid(-0.5:0.5:0.5,-0.5:0.5:0.5);
Z=gau(X,1).*gau(Y,1);

% stam(1:32,:)=0;
% stam(33:64,:)=1;


% stam=ones(64,64)*0.5;
%stam(3:8,3:8)=1;%[zeros(1,20) 0 0 0 0 0 0 0 0 0 0.5 0.5 0.5 0.5 0.5 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(1,13)] ;

%stam=filter2(ones(3,3)/9, stam);
%imshow(stam);
% stam(1,:)=0;
% stam(:,1)=0;
% stam(:,end)=0;
% stam(end,:)=0;

% stam=meshgrid(1:64);
% stam=0.5*stam+0.5*stam';
% stam=stam/max(stam(:));

%stam=stam+ones(32,32)*1;


%stam(end/2-1:end/2+1,end/2-1:end/2+1)=1;
%stam=filter2(Z*(1/sum(Z(:))),stam,'same');
% stam=filter2(ones(3,3)/9, stam);

load lena;
lena=lena/max(lena(:));
% lena_blur=filter2(ones(3,3)/9,lena,'same');
% stam=lena_blur;
stam=lena;

stm(:,:,1)=stam;%filter2( ones(5,5)/25, stam);
stm(:,:,2)=stam;%filter2( ones(5,5)/25, stam);
stm(:,:,3)=stam;%filter2( ones(5,5)/25, stam);
size(stm)

mul(1).cdata=stm(1:2:end,1:2:end,:);
mur(1).cdata=stm(1:2:end,2:2:end,:);
mll(1).cdata=stm(2:2:end,1:2:end,:);
mlr(1).cdata=stm(2:2:end,2:2:end,:);

% mul(1).cdata=zeros(2,2,2);
% mul(1).cdata(:,:,1)=[1 2;3 4];
% mul(1).cdata(:,:,2)=[1 2;3 4];
% mul(1).cdata(:,:,3)=[1 2;3 4];
% mul(2).cdata(:,:,1)=[5 6;7 8];
% mul(2).cdata(:,:,2)=[5 6;7 8];
% mul(2).cdata(:,:,3)=[5 6;7 8];

nom=4;  %Number Of Movies


close all

%movies_to_read=zeros(1,nom);
movies=cell(1,nom);
movies{1}=mul;
movies{2}=mur;
movies{3}=mll;
movies{4}=mlr;

% Y_Movies=cell(1,length(movies_to_read));

L=[];

temp_mat=[];

% for m=1:length(movies_to_read)
%     for f=1:length(movies{m})
%         Y_Movies{m}(f).cdata=im2double(movies{m}(f).cdata);
% 
%          [Y_Movies{m}(f).cdata(:,:,1), Y_Movies{m}(1).cdata(:,:,2), Y_Movies{m}(f).cdata(:,:,3)]=...
%              rgb2ycbcr(Y_Movies{m}(f).cdata(:,:,1), Y_Movies{m}(f).cdata(:,:,2), Y_Movies{m}(f).cdata(:,:,3));
%         
%         %imshow(Y_Movies{m}(f).cdata(:,:,1));    
%     end
% 
% end


for m=1:nom
    temp_mat=zeros(size(movies{m}(1).cdata(:,:,1),1),size(movies{m}(1).cdata(:,:,1),2),length(movies{m}));
    for f=1:length(movies{m})
        temp_mat(:,:,f)=movies{m}(f).cdata(:,:,1);
    end
    temp_mat=permute(temp_mat,[3,2,1]);
    L=[L transpose(temp_mat(:)) ];
end


% >> a1=kron(diag(sparse(ones(1,185472))),[1 1 1 0]);
% >> A=[a1;a1;a1;a1];
