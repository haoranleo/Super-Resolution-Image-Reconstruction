function [ A ] = Aconstruction( movies,Resol_x,Resol_y,Resol_t,T_Mat)
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
write2line=0;
A=[0 0 0];
alot=5;
for movie_num=1:num_of_movies
  
    
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
    
    for yL=1:movies_info(movie_num).Height
        lin
        for xL=1:movies_info(movie_num).Width
            for tL=1:movies_info(movie_num).NumFrames
                
                lin=lin+1;
                
                [ph(1), ph(2), ph(3)]=Ti(xL,yL,tL,movie_num,1,T_Mat);
                
                x0=round(ph(1))-(round(ph(1))-ph(1)==0.5);      % Nearest point to ph, in x-axis
                y0=round(ph(2))-(round(ph(2))-ph(2)==0.5);                
                t0=round(ph(3))-(round(ph(3))-ph(3)==0.5);
                
                for y=y0-sup_y-1:y0+sup_y+1
                    for x=x0-sup_x-1:x0+sup_x+1
                        for t=t0-sup_t-1:t0+sup_t+1
                            
                            column = 1 + (y-1)*Resol_x*Resol_t + (x-1)*Resol_t + (t-1);
                            
                            if ((column>=1) & (column<=Resol_x*Resol_t*Resol_y))
                                p=[x,y,t];
                                
                                val=B(p-ph,movie_num,T_Mat);
                                if ~(val<10^-alot)
                                    write2line=write2line+1;
                                    A(write2line,:)=[lin column val];
                                end
                            end
                            
                        end
                    end
                end
                
            end
        end
    end
    
end
