function [t_cyc, x_cyc, y_cyc]=Size_Of_Block_Cycle(movie_num,T_Mat,movie_info,Resol_t,Resol_x,Resol_y);
% The functions returns the cycle of lines for each axis. cyc=1 means all the lines are the same.
% cyc=Size_of_the_axis => No period at all. 
% works on the condition that a period consists of all the points with different values of ph-first_ph.
XSize=Resol_x          
YSize=Resol_y          
TSize=Resol_t          



first_ph=[];
[first_ph(1), first_ph(2), first_ph(3)]=Ti(1,1,1,movie_num,-1,T_Mat);

ph=[];


k=2;
t_cyc=1;
while k<=TSize
    t_cyc=t_cyc+1;
    % xL,yL,tL
    [ph(1), ph(2), ph(3)]=Ti(1,1,k,movie_num,-1,T_Mat);
    ph(3)-first_ph(3);
    if abs((ph(3)-first_ph(3))-round(ph(3)-first_ph(3)))<10^-5        % Difference between adjacent 
        yuval=0
        t_cyc=t_cyc-1
        break;
    end
    k=k+1;
end


k=2;
x_cyc=1;
while k<=XSize
    x_cyc=x_cyc+1;
    % xL,yL,tL
    [ph(1), ph(2), ph(3)]=Ti(k,1,1,movie_num,-1,T_Mat);
 %   ph(1)-first_ph(1)
    if abs((ph(1)-first_ph(1))-round(ph(1)-first_ph(1)))<10^-5        % Difference between adjacent 
        x_cyc=x_cyc-1
        break;
    end
    k=k+1;
end



k=2;
y_cyc=1;
while k<=YSize
    y_cyc=y_cyc+1;
    % xL,yL,tL
    [ph(1), ph(2), ph(3)]=Ti(1,k,1,movie_num,-1,T_Mat);
    if abs((ph(2)-first_ph(2))-round(ph(2)-first_ph(2)))<10^-5        % Difference between adjacent 
        y_cyc=y_cyc-1
        break;
    end
    k=k+1;
end