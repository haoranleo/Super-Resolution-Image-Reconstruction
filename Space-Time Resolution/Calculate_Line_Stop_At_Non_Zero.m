function [t_cyc x_cyc y_cyc]=Size_Of_Block_Cycle(movie_num,T_Mat,movie_info);

[ph(1), ph(2), ph(3)]=Ti(xL,yL,tL,movie_num,1,T_Mat);

for y=y0-sup_y-1:y0+sup_y+1
    for x=x0-sup_x-1:x0+sup_x+1
        for t=t0-sup_t-1:t0+sup_t+1
            
            column = 1 + (y-1)*Resol_x*Resol_t + (x-1)*Resol_t + (t-1);
            
            %                             if ((column>=1) & (column<=Resol_x*Resol_t*Resol_y))
            p=[x,y,t];
            
            val=B(p-ph,movie_num,T_Mat);
            if ~(val<10^-alot)
                %                                     write2line=write2line+1;
                das_line=[das_line [column;val]];
            end
            %                             end
            
        end
    end
end