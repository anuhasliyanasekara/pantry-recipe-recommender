# Pantry-recipe-recommender
Hybrid recipe recommender based on pantry ingredients

## Project Goal
Recommend recipes based on leftover pantry ingredients using a hybrid constraint-based + collaborative filtering approach.

## Problem Definition

Type of problem: Hybrid recommendation system - combining rule-based constraint filtering (ingredient matching) with collaborative filtering (predicting user preference) to produce ranked recipe suggestions.

Problem: Many people want to cook a good meal but don't have a specific recipe in mind and don't want to make a grocery run just for one dish. This project recommends recipes that are both feasible (made with ingredients the user already has) and preferred (matched to their taste).

Who this helps: Students living in dorms with limited groceries who want a quick meal before or after class, and people with busy work schedules who want to cook something different with what's currently in their kitchen - without eating the same thing every day.

## Evaluation

**Constraint-based filtering (feasibility):**
Sanity-checked manually - given a pantry list, confirm returned recipes only require ingredients the user has (or a small, defined number of missing ones).

**Collaborative filtering (preference prediction):**
- Ratings data split into training and test sets (80/20).
- During model selection, K-Fold cross-validation is used to compare different models/settings (e.g. SVD vs KNN, different numbers of latent factors) to avoid picking a model that just got lucky on one split.
- The best-performing model/settings (chosen via cross-validation) is then evaluated once on the held-out test set for a final, honest performance number.
- **Metrics used:**
  - **RMSE** - how far off predicted ratings are from actual ratings, on average.
  - **Precision@10** - of the top 10 recipes recommended, how many the user actually rated 4-5 stars. This matters more for us since the real output is a ranked list, not a single predicted number.

## Results (baseline)
- Model: SVD (surprise library), n_factors=20
- Test RMSE: 1.2116


## Experiments

We compared SVD and KNN-based collaborative filtering, being careful to test both fairly on identical data.

| Model | Data | RMSE |
|---|---|---|
| SVD (tuned: n_factors=10, reg_all=0.1) | Full dataset (1.13M ratings) | 1.2153 |
| KNNBasic | Filtered dataset (users/recipes with ≥20 ratings) | 0.9644 |
| SVD (tuned) | Filtered dataset (users/recipes with ≥20 ratings) | 0.8794 |

**Key findings:**
- Filtering to more active users and popular recipes improved performance far more than switching models - data sparsity, not model choice, was the primary bottleneck.
- Once compared fairly on identical data, SVD outperformed KNN.
- **Final model: SVD (n_factors=10, reg_all=0.1) trained on the filtered dataset, RMSE 0.8794.**


## Results (Final)

**Final model:** SVD (n_factors=10, reg_all=0.1), trained on filtered dataset (users/recipes with ≥20 ratings)

**Final test RMSE: 0.8727** (evaluated on a held-out test set, untouched during tuning)

### Experiment summary
| Model | Data | RMSE |
|---|---|---|
| SVD (default settings) | Full dataset | 1.2116 |
| SVD (tuned) | Full dataset | 1.2153 |
| KNNBasic | Filtered dataset | 0.9644 |
| SVD (tuned) | Filtered dataset (cross-validated) | 0.8794 |
| **SVD (tuned) — FINAL** | **Filtered dataset (held-out test)** | **0.8727** |

**Key takeaway:** Data sparsity was the primary bottleneck, not model choice - filtering to more active users/recipes improved RMSE far more than switching from SVD to KNN.
