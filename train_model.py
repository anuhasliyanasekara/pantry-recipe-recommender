import pandas as pd
from surprise import Dataset, Reader

interactions = pd.read_csv("data/RAW_interactions.csv")

# surprise only needs these 3 columns
interactions_small = interactions[["user_id", "recipe_id", "rating"]]

print(interactions_small.head())
print(interactions_small.shape)

# tell surprise the rating scale (1 to 5 stars)
reader = Reader(rating_scale=(1, 5))

# load into surprise's special data format
data = Dataset.load_from_df(interactions_small, reader)

print("Data loaded into surprise successfully")
print(type(data))


#split data 20% for testing and 80% for training
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print(type(trainset))
print(type(testset))
print("Training set size:", trainset.n_ratings)
print("Test set size:", len(testset))