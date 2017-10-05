function [ A ] = AconstructionBrute( movies,Resol_x,Resol_y,Resol_t,T_Mat)
%ACONSTRUCTION constructs the matrix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
%  Detailed explanation goes here

[num_of_movies movies_info]=GetMoviesInfo(movies);
lin=0;
write2line=0;
A=[0 0 0];
alot=5;
for movie_num=1:num_of_movies
    

    for yL=1:movies_info(movie_num).Height
        for xL=1:movies_info(movie_num).Width
            for tL=1:movies_info(movie_num).NumFrames
                lin=lin+1;
                column=0;

                for y=1:Resol_y
                    for x=1:Resol_x
                        for t=1:Resol_t
                            column=column+1;
                            
                            % calculation of the differance T^-1(pH)-pL
                            [TiX,TiY,TiT]=Ti(x,y,t,movie_num,-1,T_Mat);
                            point_for_Bi(1)=TiX-xL;
                            point_for_Bi(2)=TiY-yL;
                            point_for_Bi(3)=TiT-tL;
                            % calculation of the A element: Bi(pL-T^-1(pH))
                            val=Bi(point_for_Bi(1),point_for_Bi(2),point_for_Bi(3),movie_num,0);
                           
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
