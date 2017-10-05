function [num_of_movies, movies_info]=GetMoviesInfo(movies)
%movies
num_of_movies=length(movies);
for movie_num=1:num_of_movies
    movies_info(movie_num)=aviinfo( movies(movie_num).movie_name );
end