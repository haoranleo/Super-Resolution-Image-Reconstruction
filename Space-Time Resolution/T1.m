function [ Tx Ty Tt] = Ti(x,y,t,movie_num,Op)
% Ti returns transformed values of x,y,t according to the Op - either direct (+1) or inverse (-1)
% uses the global [number_of_movies]X[6] matrix 'T_Mat' 
% The i'th line consists of 6 parameters [ax,bx,ay,by,at,bt]
% which define the Ti affine transformation: Tx=ax*x+bx Ty=ay*y+by, Tt=at*t+bt.

if Op=1
    Tx=T_Mat(movie_num,1)*x+T_Mat(movie_num,2);
    Ty=T_Mat(movie_num,3)*y+T_Mat(movie_num,4);
    Tt=T_Mat(movie_num,5)*t+T_Mat(movie_num,6);
elseif Op=-1
    Tx=(x-T_Mat(movie_num,2))/T_Mat(movie_num,1);
    Ty=(y-T_Mat(movie_num,4))/T_Mat(movie_num,3)
    Tt=(t-T_Mat(movie_num,6))/T_Mat(movie_num,5)
else
    temp='invalid operation given to transform'
    Tx=[];
    Ty=[];
    Tz=[];
end