import pandas as pd
import ast
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# --- reload and prep everything needed ---
recipes = pd.read_csv("data/RAW_recipes.csv")
recipes["ingredients"] = recipes["ingredients"].apply(ast.literal_eval)

interactions = pd.read_csv("data/RAW_interactions.csv")
interactions_small = interactions[["user_id", "recipe_id", "rating"]]

user_counts = interactions_small["user_id"].value_counts()
recipe_counts = interactions_small["recipe_id"].value_counts()
active_users = user_counts[user_counts >= 20].index
popular_recipes = recipe_counts[recipe_counts >= 20].index
filtered_interactions = interactions_small[
    interactions_small["user_id"].isin(active_users) &
    interactions_small["recipe_id"].isin(popular_recipes)
]

reader = Reader(rating_scale=(1, 5))
data_small = Dataset.load_from_df(filtered_interactions, reader)
trainset_final, testset_final = train_test_split(data_small, test_size=0.2, random_state=42)

final_model = SVD(n_factors=10, reg_all=0.1, random_state=42)
final_model.fit(trainset_final)

def count_missing_ingredients(pantry, recipe_ingredients):
    missing_count = 0
    for recipe_ing in recipe_ingredients:
        matched = False
        for pantry_item in pantry:
            if pantry_item in recipe_ing:
                matched = True
                break
        if not matched:
            missing_count += 1
    return missing_count


def recommend_recipes(user_id, pantry, max_missing_ratio=0.2, top_n=10):
    """
    user_id: the user we're recommending for (used for predicted ratings)
    pantry: list of ingredients the user currently has
    max_missing_ratio: how strict the constraint filter is (0.2 = missing at most 20% of ingredients)
    top_n: how many recommendations to return
    """
    candidates = []

    # Step 1: filter by feasibility
    for index, row in recipes.iterrows():
        ingredients = row["ingredients"]
        missing = count_missing_ingredients(pantry, ingredients)
        ratio = missing / len(ingredients)

        if ratio <= max_missing_ratio:
            candidates.append(row["id"])

    # Step 2: rank candidates by predicted rating for this user
    scored_candidates = []
    for recipe_id in candidates:
        prediction = final_model.predict(uid=user_id, iid=recipe_id)
        scored_candidates.append((recipe_id, prediction.est))

    # Step 3: sort by predicted rating, highest first
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Step 4: return top N, with recipe names attached
    top_results = []
    for recipe_id, predicted_rating in scored_candidates[:top_n]:
        recipe_name = recipes.loc[recipes["id"] == recipe_id, "name"].values[0]
        top_results.append((recipe_name, predicted_rating))

    return top_results

my_pantry = ["squash", "honey", "butter", "oil", "salt", "onion", "garlic"]
results = recommend_recipes(user_id=8937, pantry=my_pantry)

for name, predicted_rating in results:
    print(f"{predicted_rating:.2f} - {name}")