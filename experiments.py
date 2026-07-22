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