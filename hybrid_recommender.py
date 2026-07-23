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
    candidates = []

    for index, row in recipes.iterrows():
        ingredients = row["ingredients"]
        missing = count_missing_ingredients(pantry, ingredients)
        ratio = missing / len(ingredients)
        num_ingredients = len(ingredients)

        if ratio <= max_missing_ratio:
            candidates.append((row["id"], row["name"], ratio, num_ingredients))

    # check if this user actually has rating history in our training data
    known_users = trainset_final.all_users()  # internal surprise IDs, not raw user_ids
    is_known_user = user_id in filtered_interactions["user_id"].values

    if not is_known_user:
        print(f"User {user_id} has no rating history — using constraint-based fallback")
        # fall back to Step 4 style ranking: missing ratio, then most ingredients
        candidates.sort(key=lambda x: (x[2], -x[3]))
        return [(name, f"ratio={ratio:.2f}") for _, name, ratio, num_ingredients in candidates[:top_n]]

    # known user — use collaborative filtering ranking
    scored_candidates = []
    for recipe_id, name, ratio, num_ingredients in candidates:
        prediction = final_model.predict(uid=user_id, iid=recipe_id)
        scored_candidates.append((name, prediction.est))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[:top_n]

my_pantry = ["squash", "honey", "butter", "oil", "salt", "onion", "garlic"]

print("=== Known user ===")
for name, score in recommend_recipes(user_id=8937, pantry=my_pantry):
    print(f"{score:.2f} - {name}")

print("\n=== Unknown/new user (like you) ===")
for name, score in recommend_recipes(user_id=99999999, pantry=my_pantry):
    print(score, "-", name)