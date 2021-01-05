# Author: Safeen Ghafour
# MovieLens Project
# HarvardX PH125.9x - Recap

# Required packages
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# Download data
if(!exists('edx')) {
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  
  # if using R 3.6 or earlier:
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  # if using R 4.0 or later:
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Validation set will be 10% of MovieLens data
  set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
}

#The RMSE funcion
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Mean of the ratings
mu <- mean(edx$rating)

#RMSE based on simple prediction
results <- tibble(method = "Average", RMSE = RMSE(validation$rating, mu))

#Movie Effect
movie_effect <- edx %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu))

#Movie effect prediction
prediction <- validation %>%
  left_join(movie_effect, by='movieId') %>%
  summarize(p = b_i + mu)

#RMSE based on movie effect
options(pillar.sigfig = 5)
results <- results %>%
  add_row(method = "Movie effect", RMSE = RMSE(validation$rating, prediction$p))

#User Effect
user_effect <- edx %>% 
  left_join(movie_effect, by='movieId') %>%
  group_by(userId) %>% 
  summarise(u_i = mean(rating - mu - b_i))

#User effect prediction  
prediction <- validation %>% 
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  mutate(p = mu + b_i + u_i) 

#RMSE based on user effect
options(pillar.sigfig = 5)
results <- results %>%
  add_row(method = "User effect", RMSE = RMSE(validation$rating, prediction$p))

#Genre Effect
genre_effect <- edx %>% 
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  group_by(genres) %>% 
  summarise(g_i = mean(rating - mu - b_i - u_i))

#Genre effect prediction  
prediction <- validation %>% 
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(genre_effect, by='genres') %>%
  mutate(p = mu + b_i + u_i + g_i) 

#RMSE based on genre effect
options(pillar.sigfig = 5)
results <- results %>%
  add_row(method = "Genre effect", RMSE = RMSE(validation$rating, prediction$p))

#Year Effect
year_effect <- edx %>% 
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(genre_effect, by='genres') %>%
  mutate(year = str_sub(title, -5, -2))  %>%
  group_by(year) %>% 
  summarise(y_i = mean(rating - mu - b_i - u_i - g_i))

#Year effects prediction  
prediction <- validation %>% 
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(genre_effect, by='genres') %>%
  mutate(year = str_sub(title, -5, -2))  %>%
  left_join(year_effect, by='year') %>%
  mutate(pred = mu + b_i + u_i + g_i + y_i) 

#RMSE based on year effect
options(pillar.sigfig = 5)
results <- results %>%
  add_row(method = "Year effect", RMSE = RMSE(validation$rating, prediction$p))

#Regularisation

#The lambda ranges, to choose an ideal one
lambdas <- seq(4, 6, 0.25)

#Loop the least RMSE
rmses <- sapply(lambdas, function(l) {
  
  #Movie Effect
  movie_effect <- edx %>% 
    group_by(movieId) %>% 
    summarise(b_i = sum(rating - mu) / (n() + l))
  
  #User Effect
  user_effect <- edx %>% 
    left_join(movie_effect, by='movieId') %>%
    group_by(userId) %>% 
    summarise(u_i = sum(rating - mu - b_i) / (n() + l))
  
  #Genre Effect
  genre_effect <- edx %>% 
    left_join(movie_effect, by='movieId') %>%
    left_join(user_effect, by='userId') %>%
    group_by(genres) %>% 
    summarise(g_i = sum(rating - mu - b_i - u_i) / (n() + l))
  
  #Year Effect
  year_effect <- edx %>% 
    left_join(movie_effect, by='movieId') %>%
    left_join(user_effect, by='userId') %>%
    left_join(genre_effect, by='genres') %>%
    mutate(year = str_sub(title, -5, -2))  %>%
    group_by(year) %>% 
    summarise(y_i = sum(rating - mu - b_i - u_i - g_i) / (n() + l))
  
  
  #All effects prediction  
  prediction <- validation %>% 
    left_join(movie_effect, by='movieId') %>%
    left_join(user_effect, by='userId') %>%
    left_join(genre_effect, by='genres') %>%
    mutate(year = str_sub(title, -5, -2))  %>%
    left_join(year_effect, by='year') %>%
    mutate(p = mu + b_i + u_i + g_i + y_i) 
  
  return(rmse = RMSE(validation$rating, prediction$p))
})

#RMSE after regularisation
options(pillar.sigfig = 5)
results <- results %>%
  add_row(method = "Regularisation effect", RMSE = min(rmses))

results
