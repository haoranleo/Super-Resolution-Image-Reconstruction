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
lin=0;
write2line=0;
A=[0 0 0];
alot=5;
for movie_num=1:num_of_movies
    lin=lin+1;

    for yL=1:movies_info(movie_num).Height
        for xL=1:movies_info(movie_num).Width
            for tL=1:movies_info(movie_num).NumFrames
                
                column=0;

                for y=1:Resol_y
                    for x=1:Resol_x
                        for t=1:Resol_t

                            column=column+1
                            p=[x,y,t];
                            ph=Ti(xL,yL,tL,movie_num,1,T_Mat);
                            val=B(p-ph,movie_num,T_Mat);
                            if ~(val<10^-alot)
                                write2line=write2line+1;
                                A(write2line,:)=[lin column val]
                            end
                            
                        end
                    end
                end
                
            end
        end
    end
    
end
