function [ interval_yH ] = IntervalCalc( y,sup_yL_maxH,sup_yL_minH,dy )
% Interval Calc -   Calculate the Interval that enters the calculation of the integral.

diff_from_max_prevy =sup_yL_maxH - (y-1);
diff_from_min_prevy =(y-1) - sup_yL_minH;                            
prevy_in_sup=(diff_from_max_prevy>0 & diff_from_min_prevy>=0);

diff_from_max_currenty =sup_yL_maxH - y;
diff_from_min_currenty =y - sup_yL_minH;                            
currenty_in_sup=(diff_from_max_currenty>0 & diff_from_min_currenty>=0);

diff_from_max_nexty =sup_yL_maxH - (y+1);
diff_from_min_nexty =(y+1) - sup_yL_minH;                            
nexty_in_sup=(diff_from_max_nexty>0 & diff_from_min_nexty>=0);

% According to the differnt possible options, the interval is calulated
%   dy for regular, mid-range point
%   fixed for points for which the interval is longer\shorter.

bool=num2str([prevy_in_sup currenty_in_sup nexty_in_sup]);
switch bool     % Check the position of the point in the support: First in the support, in the middle, last or only point in the support.
case num2str([ 0 1 1 ]),
    interval_yH=dy*(diff_from_min_currenty + 0.5);
case num2str([ 1 1 1 ]),
    interval_yH=dy;
case num2str([ 1 1 0 ]),
    interval_yH=dy*(0.5 + diff_from_max_currenty);
case num2str([ 0 1 0 ]),
    interval_yH=dy*(diff_from_min_currenty + diff_from_max_currenty);
otherwise,
    interval_yH=0;
end
