function [ Bi ] = Bi(x,y,t,movie_num,op,input)
% op=0 : returns the value of Bi(x,y,t)
% op=1 : returns a vector contains [Sup_xl_min Sup_xl_max Sup_yl_min Sup_yl_max Sup_tl_min Sup_tl_max]

if op==0

    Bi=feval(input{movie_num}.func_t,t,input{movie_num}.parameters_t).*feval(input{movie_num}.func_x,x,input{movie_num}.parameters_x).*feval(input{movie_num}.func_y,y,input{movie_num}.parameters_y);
    
elseif op==1
       Bi=[-input{movie_num}.parameters_x(1)/2,input{movie_num}.parameters_x(1)/2,-input{movie_num}.parameters_y(1)/2,input{movie_num}.parameters_y(1)/2,0,input{movie_num}.parameters_t(1)]; 
end

