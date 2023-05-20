import data_loader
from surprise import Dataset, Reader, accuracy
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import train_test_split

def funk_matrix_factorization(epochs = 10, learning_rate = 0.005, regularization = 0.02):
    """
    Trains and tests a SVD model on the user-item ratings dataset.
    """

    # Load dataframe for user-item ratings
    df = data_loader.get_user_item_ratings()

    # Load dataframe into Surprise Dataset object
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[["customer_id", "product_id", "review_score"]], reader)
    training_set, test_set = train_test_split(dataset, test_size = 0.2)

    # Train SVD on dataset
    svd = SVD(n_epochs = epochs, lr_all = learning_rate, reg_all = regularization, verbose = False)
    svd.fit(training_set)

    # Predict ratings for test set
    predictions = svd.test(test_set)

    # Compute RMSE
    rmse = accuracy.rmse(predictions)

    return svd, rmse

def hyperparameter_tuning():
    """
    Performs grid search to find optimal hyperparameters for SVD.
    Returns the best configuration, the best SVD model, and the RMSE of the best model.
    """

    epochs = [10, 20, 50, 100]
    learning_rates = [0.001, 0.005, 0.01]
    regularizations = [0.01, 0.02, 0.05, 0.1]

    svd_configs = []
    best_svd = None
    best_rmse = float("inf")
    best_index = None

    # Grid search
    index = 0
    for epoch in epochs:
        for learning_rate in learning_rates:
            for regularization in regularizations:
                svd_configs.append((epoch, learning_rate, regularization))
                
                svd, rmse = funk_matrix_factorization(epochs = epoch, learning_rate = learning_rate, regularization = regularization)
                print("Epochs: {}, Learning Rate: {}, Regularization: {}, RMSE: {}".format(epoch, learning_rate, regularization, rmse))

                if rmse < best_rmse:
                    best_svd = svd
                    best_rmse = rmse
                    best_index = index

                index += 1

    return svd_configs[best_index], best_svd, best_rmse

def generate_recommendations(svd, user_id, n = 10):
    """
    Generates recommendations for a given user ID.
    Given a (trained) SVD model and a user ID, creates a list of product IDs sorted by predicted rating.
    Returns the top n recommendations.
    """

    product_df = data_loader.get_product_dataframe()
    product_ids = product_df["product_id"].unique()

    predictions = []
    for product_id in product_ids:
        predictions.append((product_id, svd.predict(user_id, product_id).est))

    predictions.sort(key = lambda x: x[1], reverse = True)
    recommendations = [prediction[0] for prediction in predictions[:n]]

    return recommendations



def main():
    print("Training SVDs...")

    # Train SVD on dataset
    best_config, best_svd, best_rmse = hyperparameter_tuning()
    print("Best configuration: {}".format(best_config))
    print("RMSE: {}".format(best_rmse))

    # Get user IDs
    user_df = data_loader.get_user_dataframe()
    user_ids = user_df["customer_id"].unique()

    # Generate recommendations for each user
    cnt = 0
    max_cnt = len(user_ids)
    print("Generating recommendations...")
    for user_id in user_ids:
        cnt += 1
        if cnt % 100 == 0:
            print(f'{cnt}/{max_cnt}')

        recommendations = generate_recommendations(best_svd, user_id)
        
        # Write recommendations to csv file
        with open("recommendations.csv", "a") as f:
            f.write("{},".format(user_id))
            for recommendation in recommendations:
                f.write("{},".format(recommendation))
            f.write("\n")

    print("Recommendations written to recommendations.csv")



if __name__ == "__main__":
    main()