function [out_A]=Mikrei_Katze(in_A,Resol_x,Resol_y,Resol_t,T_Mat,movies_info,movie_num,input);
%HANDLING_AS_SIDES Summary of this function goes here
%  Detailed explanation goes here


false_values_pos=zeros(1,size(in_A,1));
false_values_pos(find(in_A(:,2)<1))=1;
false_values_pos(find(in_A(:,2)>Resol_x*Resol_y*Resol_t))=1;
in_A=in_A(~false_values_pos,:);


% Find which pixels of the Low_Movie may have false non-zeros elements in the appropriate A row,
% Which are not in the B support of this pixel. These pixels appear in the first place as pixels from out of the movie (H)
% which belong to B's support, during the construction of the base_block.
% Removing negative columns or columns that goes beyond the required size of A
lines_to_check=transpose(1:size(in_A,1));

row=in_A(lines_to_check,1);
column=in_A(lines_to_check,2);

% row=1 + (yL-1)*movies_info(movie_num).Width*movies_info(movie_num).NumFrames + (xL-1)*movies_info(movie_num).NumFrames + (tL-1);
% tL "=" row (mod movies_info(movie_num).NumFrames)
% xL-1 "=" (row - tL)/movies_info(movie_num).NumFrames (mod movies_info(movie_num).Width) 
% yL-1 "=" (row - tL - (xL-1)*movies_info(movie_num).NumFrames)/movies_info(movie_num).Width*movies_info(movie_num).NumFrames

tL= mod ( row, movies_info(movie_num).NumFrames) + movies_info(movie_num).NumFrames*~mod ( row, movies_info(movie_num).NumFrames);
xL=1+ mod ((row - tL)/movies_info(movie_num).NumFrames, movies_info(movie_num).Width);
yL=1+ (row - tL - (xL-1)*movies_info(movie_num).NumFrames)/(movies_info(movie_num).Width*movies_info(movie_num).NumFrames);


% column = 1 + (y-1)*Resol_x*Resol_t + (x-1)*Resol_t + (t-1)

t= mod ( column, Resol_t) + Resol_t*~mod ( column, Resol_t);
x= 1 + mod ((column - t)/Resol_t, Resol_x);
y= 1 + (column - t - (x-1)*Resol_t)/(Resol_x*Resol_t);


[TiX,TiY,TiT]=Ti(x,y,t,movie_num,-1,T_Mat);
point_for_Bi1=TiX-xL;
point_for_Bi2=TiY-yL;
point_for_Bi3=TiT-tL;
val=Bi(point_for_Bi1,point_for_Bi2,point_for_Bi3,movie_num,0,input);
%Q=in_A(find(~val),:)

out_A=in_A;
out_A(lines_to_check,3)=(~~val).*out_A(lines_to_check,3);
