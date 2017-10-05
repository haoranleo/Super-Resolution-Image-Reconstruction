mv=permute(reshape(x,Resol_t,Resol_x,Resol_y),[3 2 1]);

out_prefix='spacetime1';
fps_saved=5;

movie_out=[];
for j=1:Resol_t
    movie_out(j).cdata(:,:,1)=mv(:,:,j);
    movie_out(j).cdata(:,:,2)=mv(:,:,j);
    movie_out(j).cdata(:,:,3)=mv(:,:,j);
    movie_out(j).colormap=[];
end

eval('movname=[out_prefix];');

movie2avi(movie_out,movname,'FPS',fps_saved,'COMPRESSION','None');