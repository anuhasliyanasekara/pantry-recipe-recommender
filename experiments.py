import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

interactions = pd.read_csv("data/RAW_interactions.csv")
interactions_small = interactions[["user_id", "recipe_id", "rating"]]

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions_small, reader)

for factors in [10, 20, 50, 100]:
    print(f"\nTesting n_factors={factors}")
    model = SVD(n_factors=factors, random_state=42)
    results = cross_validate(model, data, measures=["RMSE"], cv=3, verbose=False)
    avg_rmse = results["test_rmse"].mean()
    print(f"Average RMSE: {avg_rmse:.4f}")


for reg in [0.02, 0.05, 0.1, 0.2]:
    print(f"\nTesting reg_all={reg}")
    model = SVD(n_factors=10, reg_all=reg, random_state=42)
    results = cross_validate(model, data, measures=["RMSE"], cv=3, verbose=False)
    avg_rmse = results["test_rmse"].mean()
    print(f"Average RMSE: {avg_rmse:.4f}")


# keep only users who rated at least 20 recipes, and recipes rated at least 20 times
user_counts = interactions_small["user_id"].value_counts()
recipe_counts = interactions_small["recipe_id"].value_counts()

active_users = user_counts[user_counts >= 20].index
popular_recipes = recipe_counts[recipe_counts >= 20].index

filtered_interactions = interactions_small[
    interactions_small["user_id"].isin(active_users) &
    interactions_small["recipe_id"].isin(popular_recipes)
]

print("Original size:", interactions_small.shape)
print("Filtered size:", filtered_interactions.shape)
print("Unique users:", filtered_interactions["user_id"].nunique())
print("Unique recipes:", filtered_interactions["recipe_id"].nunique())

data_small = Dataset.load_from_df(filtered_interactions, reader)


from surprise import KNNBasic

model_knn = KNNBasic(random_state=42)
results_knn = cross_validate(model_knn, data_small, measures=["RMSE"], cv=3, verbose=False)
avg_rmse_knn = results_knn["test_rmse"].mean()
print(f"KNNBasic Average RMSE: {avg_rmse_knn:.4f}")

model_svd_filtered = SVD(n_factors=10, reg_all=0.1, random_state=42)
results_svd_filtered = cross_validate(model_svd_filtered, data_small, measures=["RMSE"], cv=3, verbose=False)
avg_rmse_svd_filtered = results_svd_filtered["test_rmse"].mean()
print(f"SVD (filtered data) Average RMSE: {avg_rmse_svd_filtered:.4f}")