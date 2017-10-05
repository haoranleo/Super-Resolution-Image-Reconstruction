function [ Bi ] = Bi(x,y,t,movie_num,op)
% op=0 : returns the value of(Bi(x,y,t)
% op=1 : returns a vector contains [Sup_xl_min Sup_xl_max Sup_yl_min Sup_yl_max Sup_tl_min Sup_tl_max]
%  contains the different Bi accordingly
%  Detailed explanation goes here

expo_time=3/7;
if op==0
    %[x,y,t]
%    Bi=(1/expo_time)*rect((t-expo_time/2)/expo_time)*delta([x,y]);     %DEfined in the low resolution space of the i-th movie
%   Bi=(1/expo_time)*(1/(2.01)^2)*rect((t-expo_time/2)/expo_time).*rect(x/2.01).*rect(y/2.01); 
   Bi=(1/expo_time)*(1/(0.1)^2)*rect((t-expo_time/2)/expo_time).*rect(x/0.1).*rect(y/0.1); 
  
% Bi=(1/expo_time)*(1/(Q(-0.505/1)-Q(0.505/1)))^2*rect((t-expo_time/2)/expo_time).*gau(x/1.01,1).*gau(y/1.01,1); 

   % expo_time=expo_time(%)*Tsample(i)
%Bi=rect((t-expo_time/2)/expo_time).*trng(x/1.5).*trng(y/1.5); 
elseif op==1
%    Bi=[-1.005,1.005,-1.005,1.005,0,expo_time]; % xmin xmax
       Bi=[-0.05,0.05,-0.05,0.05,0,expo_time]; % xmin xmax

end

