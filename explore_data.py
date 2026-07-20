import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

recipes = pd.read_csv("data/RAW_recipes.csv")
interactions = pd.read_csv("data/RAW_interactions.csv")

print("RECIPES")
print(recipes.shape)
print(recipes.columns.tolist())
print(recipes.head())

print("\nINTERACTIONS")
print(interactions.shape)
print(interactions.columns.tolist())
print(interactions.head())