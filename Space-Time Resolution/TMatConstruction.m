function T_Mat=TMatConstruction(movies_names,input,Resol_x,Resol_y,Resol_t)
%TMATCONSTRUCTION Summary of this function goes here
%  Detailed explanation goes here
[num_of_movies movies_info]=GetMoviesInfo(movies_names);


T_Mat=zeros(num_of_movies,6);

for i=1:num_of_movies
    ResolRatioT(i)=Resol_t/movies_info(i).NumFrames;
    ResolRatioX(i)=Resol_x/movies_info(i).Width;
    ResolRatioY(i)=Resol_y/movies_info(i).Height;
% %[ Resolution Ratio between low space and high space(x), Shift in the low space (x)...]
    T_Mat(i,:)=[ResolRatioX(i),input{i}.shift_x,ResolRatioY(i),input{i}.shift_y,ResolRatioT(i),input{i}.shift_t];
end
