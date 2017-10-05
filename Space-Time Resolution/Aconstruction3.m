function [ A ] = Aconstruction3( movies,Resol_x,Resol_y,Resol_t,T_Mat)
%ACONSTRUCTION constructs the matrix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
%  Detailed explanation goes here

% num_of_movies=length(movies);
% for movie_num=1:num_of_movies
%     info=aviinfo(movies(movie_num));
%     duration(movie_num)=info.NumFrames;
%     width(movie_num)=info.Width;
%     height(movie_num)=info.Height;
% end

[num_of_movies movies_info]=GetMoviesInfo(movies);

for f=1:num_of_movies
    ResolRatioT(f)=Resol_t/movies_info(f).NumFrames;
    ResolRatioX(f)=Resol_x/movies_info(f).Width;
    ResolRatioY(f)=Resol_y/movies_info(f).Height;
end

lin=0;
% write2line=0;
% A=[0 0 0];

alot=5;

As=cell(num_of_movies);

for movie_num=1:num_of_movies
    
    das_line=[];

    Supports=Bi(0,0,0,movie_num,1);
    [q1, q3, q5]= Ti(Supports(1),Supports(3),Supports(5),movie_num,1,T_Mat);
    [q2, q4, q6]= Ti(Supports(2),Supports(4),Supports(6),movie_num,1,T_Mat);
    Sup_min_h=ceil([q1, q3, q5]);
    Sup_xh_min=Sup_min_h(1);
    Sup_yh_min=Sup_min_h(2);
    Sup_th_min=Sup_min_h(3);
    
    Sup_max_h=floor([q2, q4, q6]);
    Sup_xh_max = Sup_max_h(1);
    Sup_yh_max = Sup_max_h(2);
    Sup_th_max = Sup_max_h(3);
    
    sup_t = Sup_th_max - Sup_th_min + 1;
    sup_x = Sup_xh_max - Sup_xh_min + 1;
    sup_y = Sup_yh_max - Sup_yh_min + 1;

            end
        end
    end
    
    non_zero_support=find(das_line(2,:));
    first_non_zero=min(non_zero_support);
    last_non_zero=max(non_zero_support);
    das_line=das_line(:,first_non_zero:last_non_zero);
    
%    CONVOLVE2D(   [ Identity_Matrix in required height KRONICKER WITH ([1 0 .... 0] in the length of the movement required)] AND desired vector )
    As{movie_num}=
    
end









%     
%     for yL=1:1%movies_info(movie_num).Height
% %        lin
%         for xL=1:1%movies_info(movie_num).Width
%             for tL=1:1%movies_info(movie_num).NumFrames
%                 
%                lin=lin+1;
                
%                 LowInitialSample=[1 1 1; 2 1 1; 1 2 1; 1 1 2];
%                 for LIS=1:size(LowInitialSample,1)
% 
%                     [tL xL yL]=LowInitialSample(LIS,:);
%                     first_non_zero_in_line(LIS)=Calculate_Line_Stop_At_Non_Zero(xL,yL,tL,movie_num,T_Mat,sup_y,sup_x,sup_t,Resol_x,Resol_t,Resol_y,alot);
%                 end
  
                
                %   if first_non_zero_in_line(LIS) all equal all lines are identical - calculate line with 1,1,1
                 
                %   if first_non_zero_in_line(1)~=first_non_zero_in_line(4) the calculate num_frames*len_x*(until getting period)
                
                %   so => first_non_zero_in_line(1)=first_non_zero_in_line(4) the period is at most num_frames*len_x
                %       if first_non_zero_in_line(1)~=first_non_zero_in_line(3) the it is exact  num_frames*len_x. calculate this much lines
                %       else, first_non_zero_in_line(1)~=first_non_zero_in_line(2), and need to calculate only num_frames lines
                