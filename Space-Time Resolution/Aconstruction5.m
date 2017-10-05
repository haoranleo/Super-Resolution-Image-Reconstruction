function [ A ] = Aconstruction( movies,Resol_x,Resol_y,Resol_t,T_Mat)
%ACONSTRUCTION constructs the matrix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
%  Detailed explanation goes here

[num_of_movies movies_info]=GetMoviesInfo(movies);

for f=1:num_of_movies
    ResolRatioT(f)=Resol_t/movies_info(f).NumFrames;
    ResolRatioX(f)=Resol_x/movies_info(f).Width;
    ResolRatioY(f)=Resol_y/movies_info(f).Height;
end

dx=1;
dy=1;
dt=1;

lin=0;
write2line=0;
A=sparse([]);
alot=5;
for movie_num=1:num_of_movies
    
    % finding the basic cycle in each axis in the HIGH space
    % the cycle in each axis is defined as the number of DIFFERENT values that the difference pL-T^-1(pH)
    % can take where pL is a LOW resolution point and pH is a HIGH resolution point. the values of A are completely determined by the latter difference,
    % thus, the different values it recieves determines the different A-values.
    [t_cyc, x_cyc, y_cyc]=Size_Of_Block_Cycle(movie_num,T_Mat,movies_info(movie_num),Resol_t,Resol_x,Resol_y);
    
    % the basic cycles in the LOW space are related to those in the high space by the resolution ratio
    % We will be interested in finding the different rows in A (henceforth: generating rows), this is actualy assessing how many different cases of pLs exist.
    t_cycL=t_cyc/ResolRatioT(movie_num);
    x_cycL=x_cyc/ResolRatioX(movie_num);
    y_cycL=y_cyc/ResolRatioY(movie_num);
    
    NotRound(t_cycL)
    NotRound(x_cycL)
    NotRound(y_cycL)
    
    % retriving the support of Bi relative to the origin in the i'th LOW space
    Supports=Bi(0,0,0,movie_num,1);
    
    % transforming the support to the the HIGH space. NOTE: only the size of the support has meaning.
    [q1, q3, q5]= Ti(1+Supports(1),1+Supports(3),1+Supports(5),movie_num,1,T_Mat);
    [q2, q4, q6]= Ti(1+Supports(2),1+Supports(4),1+Supports(6),movie_num,1,T_Mat);
    
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% base_block calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    % We now calculate all the generating rows of the i'th movie which correspond to a set of LOW space pixels. these points
    % will generate the portion of the matrix A which corresponds to the movie by shifting and replication.
    % the set of generating points is determined by the LOW space cycles as derived from the cases of pL-T^-1(pH) (see above cycL).
    for yL=1:y_cycL
        for xL=1:x_cycL
            for tL=1:t_cycL
                [sup_xL_minH, sup_yL_minH, sup_tL_minH]= Ti(xL+Supports(1),yL+Supports(3),tL+Supports(5),movie_num,1,T_Mat);
                [sup_xL_maxH, sup_yL_maxH, sup_tL_maxH]= Ti(xL+Supports(2),yL+Supports(4),tL+Supports(6),movie_num,1,T_Mat);
                
                base_block{movie_num,tL,xL,yL}=[];      % base_block contains in {tL,xL,yL} a sparse representation of the appropriate line
                % corresponding to the point               
                [ph(1), ph(2), ph(3)]=Ti(xL,yL,tL,movie_num,1,T_Mat);
                
                x0=round(ph(1))-(round(ph(1))-ph(1)==0.5);      % Nearest point to ph, in x-axis
                y0=round(ph(2))-(round(ph(2))-ph(2)==0.5);      % same for y
                t0=round(ph(3))-(round(ph(3))-ph(3)==0.5);      % same for t
                
                % calculation of the row corresponding to the current pL=[tL,xL,yL] relative to the row 1 which is the point (1,1,1) OF THE I'TH MOVIE
                row=1 + (yL-1)*movies_info(movie_num).Width*movies_info(movie_num).NumFrames + (xL-1)*movies_info(movie_num).NumFrames + (tL-1);
                
                
                % we search for non-zero elements of A within the area of the support around the given pL
                for y=y0-sup_y-1:y0+sup_y+1
                    for x=x0-sup_x-1:x0+sup_x+1
                        for t=t0-sup_t-1:t0+sup_t+1
                            % claculation of the column corresponding to the HIGH space point pH=[t,x,y]                            
                            column = 1 + (y-1)*Resol_x*Resol_t + (x-1)*Resol_t + (t-1);
                            % calculation of the differance T^-1(pH)-pL
                            [TiX,TiY,TiT]=Ti(x,y,t,movie_num,-1,T_Mat);
                            point_for_Bi(1)=TiX-xL;
                            point_for_Bi(2)=TiY-yL;
                            point_for_Bi(3)=TiT-tL;
                            % calculation of the A element: Bi(pL-T^-1(pH))
                            val=Bi(point_for_Bi(1),point_for_Bi(2),point_for_Bi(3),movie_num,0);
                            
                            diff_from_max_x =sup_xL_maxH - x;
                            diff_from_min_x =x - sup_xL_minH;
                            
                            diff_from_max_y =sup_yL_maxH - y;
                            diff_from_min_y =y - sup_yL_minH;                            
                            
                            diff_from_max_t =sup_tL_maxH - t;
                            diff_from_min_t =t - sup_tL_minH;                            
                            
                            interval_xH=(diff_from_max_x>=(dx/2)) * dx/2 + (diff_from_max_x<(dx/2)) * diff_from_max_x+...
                                (diff_from_min_x>=(dx/2)) * dx/2 + (diff_from_min_x<(dx/2)) * diff_from_min_x;
                            if (row==1 & column==1 & val~=0)
                                sdfas=1
                            end
                            % If diff_from_max_x AND/OR diff_from_min_x are negative , x is out of B's support, and val will equal 0! no need to adjust the interval.
                            
                            interval_yH=(diff_from_max_y>=(dy/2)) * dy/2 + (diff_from_max_y<(dy/2)) * diff_from_max_y+...
                                (diff_from_min_y>=(dy/2)) * dy/2 + (diff_from_min_y<(dy/2)) * diff_from_min_y; 
                            
                            interval_tH=(diff_from_max_t>=(dt/2)) * dt/2 + (diff_from_max_t<(dt/2)) * diff_from_max_t+...
                                (diff_from_min_t>=(dt/2)) * dt/2 + (diff_from_min_t<(dt/2)) * diff_from_min_t;                                         
                            
                            val=val*interval_xH*interval_yH;%*interval_tH;
                            
                            % we keep the value only if it isn't too small
                            if ~(val<10^-alot)
                                base_block{movie_num,tL,xL,yL}=[base_block{movie_num,tL,xL,yL}; [row column val]];
                            end
                        end
                    end
                end
            end
        end
    end %% end of base_block calculation
    
    
    %%%%%%%%%%%%%%%% base_block replication %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % movie_A{movie_num,y} is a temporary structure which will eventually contain in {movie_num,y} the set of values corresponding to the y index of the generating row pL.
    % this set is attained by evaluating all the (t,x) by replication (see below)
    movie_A{movie_num,y_cycL}=[];
    
    %%%%%%%%% now creating A. for each xL(1 to x_cycL),yL, duplicate the t cycle needed number of times. Then copy the blobk created 
    %%%%%%%%% needed number of times to create the block that suits the first yL.
    %%%%%%%%% repeat this for every yL=1:y_cycL. Then copy the block created this way to get the A matrix (The part which result
    %%%%%%%%% from this specific movie)
    
    for yL=1:y_cycL
        for xL=1:x_cycL
            num_of_copies=Resol_t/t_cyc;  % the number of copies that shall be replicated along the t direction
                                          % NOTE: num_of_copies is the exact number of t-replications that will be needed (no remainder) by construction (see main)
            
            dist=t_cyc;                   % dist is the number of HIGH space points contained in a time block. adjacent (in t) time blocks
                                          % differ in column (HIGH space) by dist.
          
            % Creating the block to duplicate
            block_to_copy=[];
            for tL=1:t_cycL
                block_to_copy=[block_to_copy; base_block{movie_num,tL,xL,yL}];      
            end
            % fixing the the first index (of the rows) and the second (of the columns)
            index_fix=kron(transpose([0:num_of_copies-1]),repmat([ t_cycL dist 0],size(block_to_copy,1),1));
            
            block_to_copy=repmat(block_to_copy,num_of_copies,1);
            if length(find(size(block_to_copy)))==0         %if there's no block to copy, go on to next one
                continue;
            end
            
            temp_A=block_to_copy+index_fix; % The duplicated time block with correct row, col values.
            
            %appending the time blocks for consecutive xLs
            movie_A{movie_num,yL}=[movie_A{movie_num,yL}; temp_A];    
        end
        
        
        % Now we duplicate the "T-X-Block" we created
        num_of_copies=Resol_x/x_cyc;
        dist=Resol_t*x_cyc;             % dist is the number of HIGH space points contained in a t-x block. adjacent (in x) t-x blocks
                                        % differ in column (HIGH space) by dist.
        block_to_copy=movie_A{movie_num,yL};
        index_fix=kron(transpose([0:num_of_copies-1]), repmat( [movies_info(movie_num).NumFrames*x_cyc*(movies_info(movie_num).Width/Resol_x) dist 0] , size(block_to_copy,1) , 1)  );
        block_to_copy=repmat(block_to_copy,num_of_copies,1);
        if length(find(size(block_to_copy)))==0             %if there's no block to copy, go on to next one
            continue;
        end
                
        temp_A=block_to_copy+index_fix;% The duplicated t-x BLOCKS with correct row, col values.
        movie_A{movie_num,yL}=temp_A;
    end
    
    
    
    % Now we replicate the "T-X-Y-Block"
    num_of_copies=(Resol_y/y_cyc);
    dist=Resol_t*Resol_x*y_cyc;
    block_to_copy=[];
    for y=1:y_cyc*(movies_info(movie_num).Height/Resol_y)
        block_to_copy=[block_to_copy; movie_A{movie_num,y}];
    end
    index_fix=kron(transpose([0:num_of_copies-1]),repmat([movies_info(movie_num).NumFrames*movies_info(movie_num).Width*y_cyc*(movies_info(movie_num).Height/Resol_y) dist 0],size(block_to_copy,1),1));
    block_to_copy=repmat(block_to_copy,num_of_copies,1);
    if length(find(size(block_to_copy)))==0   % if there's no block to copy, go on to next one
        continue;
    end
    temp_A=block_to_copy+index_fix;
    
    
    
    % handeling of replication illigal artifacts
    temp_A=Mikrei_Katze(temp_A,Resol_x,Resol_y,Resol_t,T_Mat,movies_info,movie_num);
    
    % Now we should put the result in the right place, taking to account the different positions for different movies
    fix_movie_index=repmat([(movie_num>1)*sum( [(movies_info(1:movie_num-1).NumFrames)].*[(movies_info(1:movie_num-1).Height)].*[(movies_info(1:movie_num-1).Width)]) 0 0],size(temp_A,1),1);
    temp_A=temp_A+fix_movie_index;
    % And Adding to A
    A=[A; temp_A];
       
    clear temp_A
end


min_col=min(A(:,2))
max_col=max(A(:,2))

min_row=min(A(:,1))
max_row=max(A(:,1))

A=spconvert(A);


% And now taking care that the size of A will be as needed, by adding zeros on the sides, if they are missing.
if min_col>1
    A=[spalloc(size(A,1),min_col-1,1) A];
end

if max_col<Resol_x*Resol_y*Resol_t
    A=[A spalloc(size(A,1),Resol_x*Resol_y*Resol_t-max_col,1)];
end


% if min_row>1
%     A=[spalloc(min_row-1,size(A,2),1) A];
% end
% 
num_low_pixels=sum( [(movies_info(1:num_of_movies).NumFrames)].*...
    [(movies_info(1:num_of_movies).Height)].*[(movies_info(1:num_of_movies).Width)]);
if max_row<num_low_pixels
    A=[A; spalloc(num_low_pixels-max_row,size(A,2),1)];
end
% save baseblock.mat base_block;
% save baseblock.mat movie_A -append;