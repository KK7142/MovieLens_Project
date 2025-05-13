##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#################
# Splitting data 
####################

# Split edx into training (90%) and testing (10%) sets
set.seed(1)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index, ]
test_set <- edx[test_index, ]

# Ensure test_set has users/movies present in train_set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


####################
# Original Model v.1 - Predicting the average rating
##################
# Calculate the average rating in the training set
mu <- mean(train_set$rating)

# Predict the same average for all test ratings
predictions <- rep(mu, nrow(test_set))

# Calculate RMSE (how good/bad this is)
rmse_v1 <- sqrt(mean((rep(mu, nrow(test_set)) - test_set$rating)^2))
cat("\nRMSE v.1 (Baseline - Average Only):", rmse_v1, "\n")


#############
# Improved Model v.2 - Accounting for movie bias ratings
#############
# Calculate movie effects (difference from average)
movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Predict ratings: average + movie bias
predictions_v2 <- test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Handle missing values (if a movie is new)
predictions_v2 <- ifelse(is.na(predictions_v2), mu, predictions_v2)

# Calculate RMSE
rmse_v2 <- sqrt(mean((predictions_v2 - test_set$rating)^2))
cat("RMSE v.2 (Movie Effects):", rmse_v2, "\n")
rmse_v2

##################
# Further Improvements v.3 - Account for user effects
######################
# # Calculate user effects (users who rate higher or lower than average)
user_effects <- train_set %>% 
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict with movie + user effects
predictions_v3 <- test_set %>% 
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Handle NA values (new movies/users)
predictions_v3 <- ifelse(is.na(predictions_v3), mu, predictions_v3)

# Calculate RMSE
rmse_v3 <- sqrt(mean((predictions_v3 - test_set$rating)^2))
cat("RMSE v.3 (Movie + User Effects):", rmse_v3, "\n")


#######################
# Improvements v.4 - Regularization (prevention of overfitting)
#######################

# Tune lambda (penalty term) using cross-validation
lambdas <- seq(0, 10, 0.25)

# Calculate regularized movie effects
rmses <- sapply(lambdas, function(lambda){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
# Calculate regularized user effects
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
# Generate predictions
  preds <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
# Return RMSE for this lambda
  return(sqrt(mean((preds - test_set$rating)^2)))
})

# Find best lambda (minimum RMSE)
lambda <- lambdas[which.min(rmses)]
lambda  # Print optimal lambda value

# Train final regularized model with best lambda
b_i_reg <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- train_set %>%
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Make predictions
predictions_v4 <- test_set %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Handle NA values (new movies/users)
predictions_v4 <- ifelse(is.na(predictions_v4), mu, predictions_v4)

# Calculate RMSE
rmse_v4 <- sqrt(mean((predictions_v4 - test_set$rating)^2))
cat("RMSE v.4 (Regularized):", rmse_v4, "\n")

#############
# v.5 - Final Holdout Test (ONE-TIME EVALUATION of final_holdout_test set)
##############
# Train final model on FULL edx data
mu_final <- mean(edx$rating)

# Use pre-tuned lambda value from v.4
lambda_final <- lambda

# Calculate regularized movie effects on FULL edx data
b_i_final <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu_final)/(n() + lambda))  # Use optimal lambda

# Calculate regularized user effects on FULL edx data
b_u_final <- edx %>%
  left_join(b_i_final, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_final)/(n() + lambda))

# Make predictions on final holdout set
predictions_v5 <- final_holdout_test %>% 
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  mutate(pred = mu_final + b_i + b_u) %>%
  pull(pred)

# Handle NAs and calculate RMSE
predictions_v5 <- ifelse(is.na(predictions_v5), mu_final, predictions_v5)

# Calculate final RMSE v.5
rmse_v5 <- sqrt(mean((predictions_v5 - final_holdout_test$rating)^2))
cat("RMSE FINAL HOLDOUT TEST v.5:", rmse_v5, "\n")

#############
# Final Model Comparison Summary
##############
# Final Summary (using existing RMSE values)
cat("\nMODEL IMPROVEMENT PROGRESS:\n",
    "----------------------------------------\n",
    "v.1 (Baseline):          ", round(rmse_v1, 5), "\n",
    "v.2 (+Movie Effects):    ", round(rmse_v2, 5), "\n", 
    "v.3 (+User Effects):     ", round(rmse_v3, 5), "\n",
    "v.4 (Regularized):       ", round(rmse_v4, 5), "\n",
    "v.5 (Final Holdout Test):", round(rmse_v5, 5), "\n",
    "----------------------------------------\n")

# Table For R Markdown report
model_results <- data.frame(
  "Model Version" = c("1. Baseline", 
                      "2. + Movie Effects", 
                      "3. + User Effects", 
                      "4. Regularized",
                      "5. Final Test"),
  RMSE = c(rmse_v1, rmse_v2, rmse_v3, rmse_v4, rmse_v5),
  Improvement = c("--", 
                  rmse_v1 - rmse_v2, 
                  rmse_v2 - rmse_v3, 
                  rmse_v3 - rmse_v4,
                  rmse_v4 - rmse_v5)
)

# Basic table output
print(model_results)

###############
# Visuals for Data Explortation for Report
######################

# Distribution of Ratings
ggplot(edx, aes(x = rating)) + 
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
  ggtitle("Distribution of Movie Ratings") +
  xlab("Rating") + ylab("Count")

# Ratings Per Movie
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, fill = "darkgreen") +
  scale_x_log10() + 
  ggtitle("Number of Ratings Per Movie") +
  xlab("Number of Ratings (log scale)") + ylab("Count")

# Ratings Per User
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, fill = "violet") +
  scale_x_log10() + 
  ggtitle("Number of Ratings Per User") +
  xlab("Number of Ratings (log scale)") + ylab("Count")

# RMSE Comparison Plot
rmse_results <- data.frame(
  Version = factor(c("v.1", "v.2", "v.3", "v.4"), 
                   levels = c("v.1", "v.2", "v.3", "v.4")),
  RMSE = c(rmse_v1, rmse_v2, rmse_v3, rmse_v4)
)

ggplot(rmse_results, aes(x = Version, y = RMSE, fill = Version)) +
  geom_col() +
  geom_text(aes(label = round(RMSE, 5)), vjust = -0.5) +
  ggtitle("RMSE by Model Version") +
  ylim(0, max(rmse_results$RMSE) * 1.1) +
  theme_minimal()

