import pandas as pd
import ast

recipes = pd.read_csv("data/RAW_recipes.csv")

# apply ast.literal_eval to every row in the 'ingredients' column(Make the column of strings into a list)
recipes["ingredients"] = recipes["ingredients"].apply(ast.literal_eval)

# sanity check
print(type(recipes.loc[0, "ingredients"]))  # should say <class 'list'>
print(recipes.loc[0, "ingredients"])
print(type(recipes.loc[5, "ingredients"]))  # check a different row too


all_ingredients = []
for ingredient_list in recipes["ingredients"]:
    all_ingredients.extend(ingredient_list)

unique_ingredients = set(all_ingredients)

print("Total ingredient mentions:", len(all_ingredients))
print("Unique ingredient strings:", len(unique_ingredients))

# look at all unique ingredients containing the word 'tomato'
tomato_variants = []

for ing in unique_ingredients:
    if "tomato" in ing:
        tomato_variants.append(ing)
print(tomato_variants)

#checking for recipes that user can make with his pantry

def recipe_is_makeable(pantry, recipe_ingredients):
    """
    pantry: list of ingredients the user has, e.g. ["tomato", "onion", "egg"]
    recipe_ingredients: list of ingredients a recipe needs
    Returns True if every recipe ingredient can be matched to something in the pantry.
    """
    for recipe_ing in recipe_ingredients:
        matched = False
        for pantry_item in pantry:
            if pantry_item in recipe_ing:   # substring check
                matched = True
                break   # found a match, no need to keep checking this recipe_ing
        if not matched:
            return False   # this recipe needs something the pantry doesn't have
    return True   # every ingredient found a match

#this counts how many ingrediants user has

def count_missing_ingredients(pantry, recipe_ingredients):
    """
    pantry: list of ingredients the user has
    recipe_ingredients: list of ingredients a recipe needs
    Returns the number of recipe ingredients that have NO match in the pantry.
    """
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


#give a ratio of missing items and items in the recipe
def missing_ratio(pantry, recipe_ingredients):
    """
    Returns the fraction of recipe ingredients NOT found in the pantry.
    Lower is better. 0.0 = fully makeable.
    """
    missing = count_missing_ingredients(pantry, recipe_ingredients)
    return missing / len(recipe_ingredients)


test_pantry = ["squash", "butter", "honey", "oil", "salt"]

first_recipe_ingredients = recipes.loc[0, "ingredients"]
print(first_recipe_ingredients)

result = recipe_is_makeable(test_pantry, first_recipe_ingredients)
print("Makeable?", result)

full_pantry = ["squash", "mexican seasoning", "mixed spice", "honey", "butter", "oil", "salt"]

result2 = recipe_is_makeable(full_pantry, first_recipe_ingredients)
print("Makeable with full pantry?", result2)

#Try with the whole data set(231,637)

my_pantry = ["squash", "honey", "butter", "oil", "salt", "onion", "garlic"]

makeable_recipes = []

for index, row in recipes.iterrows():
    if recipe_is_makeable(my_pantry, row["ingredients"]):
        makeable_recipes.append(row["name"])

print("Number of makeable recipes:", len(makeable_recipes))
print(makeable_recipes[:10])   # just show the first 10



#show recipe with low ratio and higher items
my_pantry = ["squash", "honey", "butter", "oil", "salt", "onion", "garlic"]

recipe_scores = []

for index, row in recipes.iterrows():
    ratio = missing_ratio(my_pantry, row["ingredients"])
    num_ingredients = len(row["ingredients"])
    recipe_scores.append((row["name"], ratio, num_ingredients))

# sort by: ratio ascending (fewer missing = better),
# then num_ingredients descending (more complete meals first)
recipe_scores.sort(key=lambda x: (x[1], -x[2]))

for name, ratio, num_ingredients in recipe_scores[:10]:
    print(f"{ratio:.2f} missing ratio, {num_ingredients} ingredients - {name}")