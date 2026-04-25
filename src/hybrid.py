import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error


# LOAD DATA

movies = pd.read_csv('../data/movies.csv')
ratings = pd.read_csv('../data/ratings.csv')


# COLLABORATIVE FILTERING

user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

user_similarity = cosine_similarity(user_item)


# CONTENT-BASED FILTERING

movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

content_similarity = cosine_similarity(tfidf_matrix)


# POPULARITY BASED (COLD START)
def popular_movies(n=5):
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    })

    movie_stats.columns = ['avg_rating', 'num_ratings']

    # Filter movies with enough ratings
    movie_stats = movie_stats[movie_stats['num_ratings'] > 50]

    # Sort by rating + popularity
    movie_stats = movie_stats.sort_values(by='avg_rating', ascending=False)

    return movies[movies['movieId'].isin(movie_stats.head(n).index)]['title']

# COLLABORATIVE RECOMMENDATION

def collaborative_recommend(user_id, n=5):
    similar_users = list(enumerate(user_similarity[user_id - 1]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = set()

    for user, _ in similar_users:
        user_movies = ratings[ratings['userId'] == user + 1]['movieId']
        recommended_movies.update(user_movies)

    return movies[movies['movieId'].isin(list(recommended_movies))]['title'].head(n)


# CONTENT-BASED RECOMMENDATION

def content_recommend(title, n=5):
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(content_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in scores]
    return movies['title'].iloc[movie_indices]


# HYBRID RECOMMENDATION

def hybrid_recommend(user_id, title, n=5):
    collab = set(collaborative_recommend(user_id, n))
    content = set(content_recommend(title, n))

    combined = list(collab.union(content))
    return combined[:n]


# SMART RECOMMENDER (COLD START HANDLING)

def smart_recommend(user_id=None, title=None, n=5):
    # New user → no data
    if user_id is None:
        print("Cold start: showing popular movies")
        return popular_movies(n)

    # If both available → hybrid
    if user_id is not None and title is not None:
        return hybrid_recommend(user_id, title, n)

    # If only user → collaborative
    if user_id is not None:
        return collaborative_recommend(user_id, n)

    # fallback
    return popular_movies(n)


# EVALUATION (RMSE)
def evaluate():
    y_true = ratings['rating']
    y_pred = np.full_like(y_true, y_true.mean())

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("Baseline RMSE:", rmse)

# MAIN

if __name__ == "__main__":
    print("\nHybrid Recommendation")
    print(smart_recommend(user_id=1, title="Toy Story (1995)"))

    print("\n Cold Start (New User)")
    print(smart_recommend())

    print("\n Evaluation")
    evaluate()