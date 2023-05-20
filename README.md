## Approach

For this recommender system, I have chosen to create a model-based, collaborative filtering recommender. The application works with Pandas dataframes and with the Surprise sk-learn package to learn and predict good recommendations for customers. Specifically, it uses the Funk Matrix Factorization algorithm, as provided by Surprise. The system trains SVD predictors, and then generates a csv file containing predictions for each user. By default, the application generates 10 predictions per user, and has a threshold of 4 for predictions that are included.

Initially, an approach using the Truncated SVD factorization was attempted. However, due to the large size of the datasets, this was infeasible, as a user-item interaction matrix must be created - this matrix is a frequency matrix for all combinations of users (customers) and items (products), which would be enormous. Funk matrix factorization has the advantage that it works directly with a rating matrix, i.e., a matrix with a row for each review given by a customer. In our case, each user leaves only a small amount of reviews (or potentially none), and as such, the user-item interaction matrix is (extremely) sparse, resulting in large amounts of useless computation.

## System

The application consists of 2 main files: data_loader.py and funk_matrix_factorization.py. The first one contains a series of utility functions which create custom data frames from the provided data files. Note: the data files are located in the sub-directory "data/" under the root folder. Some examples of data frames are the "order item" dataframe, which extracts the order ID and product ID from the rows in the CSV file, and the "rating" data frame which extracts the review ID, the order ID and the review score. The "get_user_item_ratings" returns a data frame which merges several dataframes - this is used to generate the dataset for the recommendation algorithm, as described above.

The funk_matrix_factorization function in the other file is a straight-forward function, which implements Funk matrix factorization. It calls the data_loader file to create relevant data frames, and creates a Surprise dataset from it. The machine learning algorithm is then trained, and its performance is measured before the predictor is returned. The Funk algorithm is trained using Stochastic Gradient Descent, and as such there are some hyperparameters that must be chosen - training epochs, learning rate and regularization, among others. These are the only ones we are concerned with for now, although others can be experimented with for further analysis. A hyperparameter-tuning function has been implemented, which performs a straight-forward Grid Search. The best model and configuration is returned. This tuning is optional, and should be called with the command line argument --tune.

After the best model has been found, predictions are generated for each user. Using a threshold value (default 4 out of 5), 10 predictions for each user is generated. These are written to a CSV file which saves to the root folder.

## Hyperparameter Tuning

A grid search was performed over the three relevant hyperparameters - number of epochs, learning rate and regularization. The outcome showed little variance among the choices, with final Root Mean Square Error ranging from 1.3 to approximately 1.35. The sole exception to this was the number of epochs, which is to be expected, where more epochs typically resulted in a lower error. However, 100 training epochs sometimes resulted in worse performance than 50, likely due to overfitting.

## Limitations

The very large size of the datasets impose great restrictions. As mentioned in the first section above, an approach using truncated SVD was infeasible, as generating the user-item interaction matrix would be very costly and time-consuming. 

Training with the Funk algorithm is very efficient,and runs only in a few seconds for a model. However, generating predictions is still costly and time-consuming. For this project, a simple iterative approach was chosen, where one user at a time has its predictions generated and written to a CSV output file. With more time and greater computing resources than were available for this project, vectorization and batch predictions could be investigated, to further reduce computational load. Initially, this iterative approach would take 10 hours to generate predictions for all users. For this reason, a threshold value was introduced to speed up computation, where a product would be accepted if its predicted review score was above this threshold - the default value was chosen to be 4, as other values did not significantly improve execution time. With this threshold, recommendations for all users would be generated in 2 hours, down from 10. This is a trade-off between computational load and model quality, as is always the case in Machine Learning. With further time and compute, other more sophisticated approaches could be investigated. One idea is to filter out product categories which are irrelevant to the user.

Due to the time restriction of this project, there was no time to implement data analysis on the most recommended products.

## User guide

To run this code and generate recommendations, execute the file funk_matrix_factorization.py. Optionally, a hyper-parameter tuning can be included with the (boolean) command line argument --tune. The outputs save to a recommendations.csv file in the root directory. Make sure that dependencies are installed. These may include Pandas and the sk-learn Surprise package, among others, which should be available to install with pip.

## Expected run-times
The hyper-parameter tuning and training times should take a few minutes in total, roughly 2-3. Prediction generation should predict 500 users per 30 seconds, or equivalently, ~2 hours for all 99k users. This is based on a 2018 MacBook Air, and may vary between systems.
