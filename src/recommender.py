import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('../data/movies.csv')
ratings = pd.read_csv('../data/ratings.csv')

# Create user-item matrix
user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# Compute similarity
user_similarity = cosine_similarity(user_item)
# Recommend movies
def recommend_movies(user_id, num_recommendations=5):
    similar_users = list(enumerate(user_similarity[user_id - 1]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = set()

    for user, score in similar_users:
        user_movies = ratings[ratings['userId'] == user + 1]['movieId']
        recommended_movies.update(user_movies)

    return movies[movies['movieId'].isin(list(recommended_movies))]['title'].head(num_recommendations)

# Run
if __name__ == "__main__":
    print("Recommended Movies:")
    print(recommend_movies(1, num_recommendations=5))