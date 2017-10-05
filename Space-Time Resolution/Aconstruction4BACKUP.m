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
A=sparse([]);
alot=5;
for movie_num=1:num_of_movies
    movie_num
    
    [t_cyc, x_cyc, y_cyc]=Size_Of_Block_Cycle(movie_num,T_Mat,movies_info(movie_num));
    
   
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
    
    
    for yL=1:y_cyc  %movies_info(movie_num).Height
        %         lin
        for xL=1:x_cyc  %movies_info(movie_num).Width
            for tL=1:t_cyc  %movies_info(movie_num).NumFrames
                
                base_block{movie_num,tL,xL,yL}=[];    % base_block contains in {tL,xL,yL} a sparse representation of the appropriate line
                                            % corresponding to the point 
                %                 lin=lin+1;
                
                [ph(1), ph(2), ph(3)]=Ti(xL,yL,tL,movie_num,1,T_Mat);
                
                x0=round(ph(1))-(round(ph(1))-ph(1)==0.5);      % Nearest point to ph, in x-axis
                y0=round(ph(2))-(round(ph(2))-ph(2)==0.5);                
                t0=round(ph(3))-(round(ph(3))-ph(3)==0.5);
                
                row=1 + (yL-1)*movies_info(movie_num).Width*movies_info(movie_num).NumFrames + (xL-1)*movies_info(movie_num).NumFrames + (tL-1);
                            
                for y=y0-sup_y-1:y0+sup_y+1
                    for x=x0-sup_x-1:x0+sup_x+1
                        for t=t0-sup_t-1:t0+sup_t+1
                            
                            
                            column = 1 + (y-1)*Resol_x*Resol_t + (x-1)*Resol_t + (t-1);
                            
                            %if ((column>=1) & (column<=Resol_x*Resol_t*Resol_y))
                            p=[x,y,t];
                            
                            %stm=Ti(x,y,t,movie_num,-1,T_Mat)-[xL,yL,tL]
                            [TiX,TiY,TiT]=Ti(x,y,t,movie_num,-1,T_Mat);
                            point_for_Bi(1)=TiX-xL;
                            point_for_Bi(2)=TiY-yL;
                            point_for_Bi(3)=TiT-tL;
                            %                            [point_for_Bi(1),point_for_Bi(2),point_for_Bi(3)]=[point_for_Bi(1),point_for_Bi(2),point_for_Bi(3)]-[xL,yL,tL];
                            val=Bi(point_for_Bi(1),point_for_Bi(2),point_for_Bi(3),movie_num,0);
                            if ~(val<10^-alot)
                                [x,y,t];
                                [TiX,TiY,TiT];
                                [xL,yL,tL];
                                [point_for_Bi(1),point_for_Bi(2),point_for_Bi(3)];


                                %b=[column val]
                                %[x,y,t]
                                %Ti(x,y,t,movie_num,-1,T_Mat)-[xL,yL,tL]
                                %[dx dy dt]=Ti(xc(1),xc(2),xc(3),movie_num,-1,T_Mat);

                                base_block{movie_num,tL,xL,yL}=[base_block{movie_num,tL,xL,yL}; [row column val]];
                                %         write2line=write2line+1;
                                %         A(write2line,:)=[lin column val];
                            end
                            %end
                        end
                    end
                end
                
            end
        end
    end
    
    
    base_block_size=size(base_block(:,:,:,:))
    base_block
    movie_A{movie_num,y_cyc}=[];
    
    %%%%%%%%% now crating A. for each x0(1 to x_cyc),y0, duplicate the t cycle needed number of times. Then copy the blobk created 
    %%%%%%%%% needed number of times to create the block that suits the first y0.
    %%%%%%%%% repeat this for every y0=1:y_cyc. Then copy the block created this way to get the A matrix (The part which result
    %%%%%%%%% from this specific movie)
    
    for y=1:y_cyc   %base_block_size(4)
        for x=1:x_cyc   %base_block_size(3)
            num_of_copies=ceil(movies_info(movie_num).NumFrames/t_cyc)  % to duplicate the t_cycle
            remainder=mod(movies_info(movie_num).NumFrames,t_cyc)   % If it's not a whole number
            
            dist=(Resol_t/movies_info(movie_num).NumFrames)*t_cyc;  % shifting to the right of the duplicated blocks
                            % Getting down a rows will require shifting to the right of a*Resol_t/movies_info(movie_num).NumFrames
                            % However we duplicate blocks of t_cyc rows. NOTE: THis number should result in a whole number
                            
                            % Make sure, for example, that Width*ResolRationinX = whole number!!
            
            block_to_copy=[];
            for t=1:base_block_size(2)
                block_to_copy=[block_to_copy; base_block{movie_num,t,x,y}]      % Creating the block to duplicate
            end
            
            index_fix=kron(transpose([0:num_of_copies-1]),repmat([t_cyc dist 0],size(block_to_copy,1),1));
                        % We fix the first index (of the rows) and the second (of the columns)
            block_to_copy=repmat(block_to_copy,num_of_copies,1);
            
            if length(find(size(block_to_copy)))==0         % ???????? if there's no block to copy, go on to next one
                continue;
            end
            
            temp_A=block_to_copy+index_fix; % The duplicated block
            
            % If remainder was not 0, we now have to remove some lines from the end
            lines_ind_remove=[];
            
            lines_ind_remove=[lines_ind_remove ~~remainder*([(num_of_copies-1)*t_cyc+remainder+1:num_of_copies*t_cyc])];
            
            num_of_lines_to_remove=~~remainder*length(find(temp_A(:,1)>=lines_ind_remove(1)));
            
            % And saving
            movie_A{movie_num,y}=[movie_A{movie_num,y}; temp_A( 1 : (size(temp_A,1)-num_of_lines_to_remove) , : )];

        end
        
        
        % Now we duplicate the NumFrames*x_cyc rows block we created
        num_of_copies=ceil(movies_info(movie_num).Width/x_cyc);
        remainder=mod(movies_info(movie_num).Width,x_cyc);
        
        dist=Resol_t*x_cyc; %(Resol_t/movies_info(movie_num).NumFrames)*t_cyc;
                % This is the shifting to the right that should go with getting down NumFrames*x_cyc rows
        block_to_copy=movie_A{movie_num,y};
        
        %         block_to_copy=[];
        %         for t=1:base_block_size(2)
        %             block_to_copy=[block_to_copy; base_block{movie_num,t,x,y}];
        %         end
        
        index_fix=kron(transpose([0:num_of_copies-1]),repmat([movies_info(movie_num).NumFrames*x_cyc dist 0],size(block_to_copy,1),1));
        block_to_copy=repmat(block_to_copy,num_of_copies,1);
        
        if length(find(size(block_to_copy)))==0         % ???????? if there's no block to copy, go on to next one
            continue;
        end
        
        
        temp_A=block_to_copy+index_fix;
        
        lines_ind_remove=[];
        lines_ind_remove=[ lines_ind_remove ~~remainder*( [ ((num_of_copies-1)*x_cyc+remainder)*Resol_t+1:num_of_copies*x_cyc*Resol_t])];
        
        num_of_lines_to_remove=~~remainder*length(find(temp_A(:,1)>=lines_ind_remove(1)));
        
        movie_A{movie_num,y}=temp_A( 1 : (size(temp_A,1)-num_of_lines_to_remove) , : );
        
        %movie_A{movie_num,y}=[movie_A{movie_num,y}; temp_A( 1 : (size(temp_A,1)-num_of_lines_to_remove) , : )];
        
        
    end
    
    
    
    % Same for Y
    
    
    num_of_copies=ceil(movies_info(movie_num).Height/y_cyc)
    remainder=mod(movies_info(movie_num).Height,y_cyc)
    
    dist=Resol_t*Resol_x*y_cyc; %(Resol_t/movies_info(movie_num).NumFrames)*t_cyc;
    
    %        block_to_copy=movie_A{movie_num,y};
    
    block_to_copy=[];
    for t=1:y_cyc
        block_to_copy=[block_to_copy; movie_A{movie_num,y}];
    end
    
    index_fix=kron(transpose([0:num_of_copies-1]),repmat([movies_info(movie_num).NumFrames*movies_info(movie_num).Width*y_cyc dist 0],size(block_to_copy,1),1));
    block_to_copy=repmat(block_to_copy,num_of_copies,1);
    
    
    if length(find(size(block_to_copy)))==0         % ???????? if there's no block to copy, go on to next one
        continue;
    end
    
    temp_A=block_to_copy+index_fix;
    
    lines_ind_remove=[];
    lines_ind_remove=[lines_ind_remove ~~remainder*( [ ((num_of_copies-1)*y_cyc+remainder)*Resol_t*Resol_y+1:num_of_copies*y_cyc*Resol_t*Resol_x])];
    
    num_of_lines_to_remove=~~remainder*length(find(temp_A(:,1)>=lines_ind_remove(1)));
    
    movie_A{movie_num,y}=temp_A( 1 : (size(temp_A,1)-num_of_lines_to_remove) , : );
    
    %movie_A{movie_num,y}=[movie_A{movie_num,y}; temp_A( 1 : (size(temp_A,1)-num_of_lines_to_remove) , : )];
    
    movie_A
    
    % Now we should put the result in the right place, taking to account the different positions for different movies
    fix_movie_index=repmat([(movie_num>1)*sum( [(movies_info(1:movie_num-1).NumFrames)].*[(movies_info(1:movie_num-1).Height)].*[(movies_info(1:movie_num-1).Width)]) 0 0],size(movie_A{movie_num},1),1);
    movie_A{movie_num}=movie_A{movie_num}+fix_movie_index;
    
    % And Adding to A
    A=[A; movie_A{movie_num}];
    
    clear movie_A
end


% Removing negative columns or columns that goes beyond the required size of A
false_values_pos=zeros(1,size(A,1));
false_values_pos(find(A(:,2)<1))=1;
false_values_pos(find(A(:,2)>Resol_x*Resol_y*Resol_t))=1;
A=A(~false_values_pos,:);

min_col=min(A(:,2));
max_col=max(A(:,2));

min_row=min(A(:,1));
max_row=max(A(:,1));

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
% if max_col<Resol_x*Resol_y*Resol_t
%     A=[A spalloc(size(A,1),Resol_x*Resol_y*Resol_t-max_col,1)];
% end
% save baseblock.mat base_block;
% save baseblock.mat movie_A -append;