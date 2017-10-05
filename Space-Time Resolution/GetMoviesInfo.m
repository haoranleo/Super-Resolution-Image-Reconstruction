function [num_of_movies, movies_info]=GetMoviesInfo(movies_names)
% Reads the info from the movies, names of which are found in the vector movies_names
num_of_movies=length(movies_names);
for movie_num=1:num_of_movies
    movies_info(movie_num)=aviinfo( movies_names{movie_num} );
end