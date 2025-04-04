# Collaborative Filtering Recommendation System based on Movie Ratings

## Overview
This project implements User-Based Collaborative Filtering (UserCF) and Item-Based Collaborative Filtering (ItemCF) recommendation systems using the MovieLens dataset. The system generates personalized movie recommendations by analyzing user ratings and calculating similarity metrics. Key features include cold-start handling, similarity matrix optimization, and evaluation insights.

**Author**: GDUFE cclear116  
**Version**: 0.3  
**Date**: 2025/3/29 - 2025/4/3  

---

## Features
### UserCF
- **Similarity Calculation**: Pearson correlation coefficient for user similarity.
- **Cold-Start Handling**: Recommends globally popular movies for new/unrated users.
- **Optimizations**: 
  - Inverted indexing for efficient candidate collection.
  - Weighted scoring using similarity and normalized ratings.

### ItemCF
- **Similarity Calculation**: Cosine similarity for item similarity.
- **Popularity Penalty**: Reduces bias toward popular items.
- **Matrix Truncation**: Retains top-20 similar items to reduce computation.
- **Cold-Start Handling**: Falls back to global popular items for new users.

### General
- **Data Preprocessing**: Handles missing values and duplicates.
- **Evaluation**: Supports MAE/RMSE and top-N metrics (accuracy, recall).

---

## Dataset
**Source**: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)  
- **Size**: 100,836 ratings from 610 users on 9,742 movies.  
- **Files**: `ratings.csv`, `movies.csv`.  

---

## Requirements
- Python 3.8+
- Libraries:
  ```bash
  pandas >= 1.3.5
  numpy >= 1.21.2


---

## Usage

### 1. Data Preparation

Place `ratings.csv` and `movies.csv` in the project directory.

### 2. Run UserCF

```python
# Generate recommendations for target users
target_users = [1, 15, 99999]
for user_id in target_users:
    recommendations = generate_recommendations(user_id)
    print(f"Recommendations for User {user_id}: {recommendations}")
```

### 3. Run ItemCF

```python
# Example recommendation for a user
user_id = 1
recommended_movies = recommend_items(user_id)
print(f"Top recommendations for User {user_id}:")
for movie, score in recommended_movies:
    print(f"- {movie} (Score: {score:.2f})")
```

---

## Example Output

### UserCF

```
=== Recommendations for User 1 ===
1. Shawshank Redemption, The (1994)
2. Terminator 2: Judgment Day (1991)
3. Apollo 13 (1995)
4. Lord of the Rings: The Fellowship of the Ring, The (2001)
5. Godfather, The (1972)
```

### ItemCF

```
User 1's Recommendations:
- Ferris Bueller's Day Off (1986) (Score: 0.83)
- Die Hard (1988) (Score: 0.81)
- Mars Attacks! (1996) (Score: 0.80)
```

---

## Optimization Insights

- **UserCF**: Achieved 1.5s runtime for 50 similar users using inverted indexing.
- **ItemCF**: Reduced matrix size by 60% with top-20 similarity truncation.
- **Cold-Start**: Global popularity ranking improved recommendation diversity.

---

## References

1. [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
2. [Collaborative Filtering Algorithms (ACM)](https://dl.acm.org/doi/10.1145/371920.372071)
3. [recommenderlab Framework](https://www.researchgate.net/publication/237246291_recommenderlab_A_Framework_for_Developing_and_Testing_Recommendation_Algorithms)

---

**Note**: For advanced usage (e.g., evaluation metrics or hybrid models), refer to the Jupyter notebook comments and cited literature.

