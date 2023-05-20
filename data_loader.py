import pandas as pd
import numpy as np


def get_user_item_ratings(filenames = ["data/olist_order_reviews_dataset.csv", "data/olist_orders_dataset.csv", "data/olist_order_items_dataset.csv"]):
    
    # Read relevant columns from reviews dataset
    reviews_df = get_rating_dataframe(filename=filenames[0])
    orders_df = get_order_dataframe(filename=filenames[1])
    order_items_df = get_order_item_dataframe(filename=filenames[2])

    # Merge relevant columns from all datasets
    df = reviews_df.merge(orders_df, on = "order_id")
    df = df.merge(order_items_df, on = "order_id")

    # Drop duplicate rows
    df = df.drop_duplicates(subset = ["customer_id", "product_id"])

    # Drop irrelevant columns
    df = df.drop(columns = ["order_id", "review_id"])

    return df


### Utility functions ###

def get_rating_dataframe(filename = "data/olist_order_reviews_dataset.csv"):
    
    # Read relevant columns from reviews dataset
    reviews_df = pd.read_csv(filename)
    reviews_df = reviews_df[["review_id", "order_id", "review_score"]]
    return reviews_df

def get_order_dataframe(filename = "data/olist_orders_dataset.csv"):

    # Read relevant columns from orders dataset
    orders_df = pd.read_csv(filename)
    orders_df = orders_df[["order_id", "customer_id"]]
    return orders_df

def get_order_item_dataframe(filename = "data/olist_order_items_dataset.csv"):

    # Read relevant columns from order items dataset
    order_items_df = pd.read_csv(filename)
    order_items_df = order_items_df[["order_id", "product_id"]]
    return order_items_df

def get_product_dataframe(filename = "data/olist_products_dataset.csv"):

    # Read relevant columns from products dataset
    products_df = pd.read_csv(filename)
    products_df = products_df[["product_id", "product_category_name"]]
    return products_df

def get_user_dataframe(filename = "data/olist_customers_dataset.csv"):

    # Read relevant columns from customers dataset
    customers_df = pd.read_csv(filename)
    customers_df = customers_df[["customer_id", "customer_unique_id"]]
    return customers_df




############ DEPRECATED ############
# def get_user_item_matrix(filenames = ["data/olist_order_reviews_dataset.csv", "data/olist_orders_dataset.csv", "data/olist_order_items_dataset.csv"]):
#     """
#     Warning: This function takes a long time to run if the dataset is large.
#     Use with caution.
#     """
    
#     # Get user-item ratings
#     df = get_user_item_ratings(filenames = filenames)

#     # Create user-item matrix
#     user_item_matrix = df.pivot_table(index = "customer_id", columns = "product_id", values = "review_score")
#     return user_item_matrix
############ DEPRECATED ############

